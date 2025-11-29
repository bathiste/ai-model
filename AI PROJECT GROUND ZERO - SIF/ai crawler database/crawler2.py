import requests
import sqlite3
import time
import nltk
import queue
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from bs4 import BeautifulSoup
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event

nltk.download('punkt', quiet=True)

# --------- Config ----------
DB = "dataset.db"
INPUT_FILE = "crawler_topicinputs.txt"
USER_AGENT = {"User-Agent": "Mozilla/5.0 (DatasetBuilder/3.0-threaded)"}
MAX_THREADS = 220                     # tuned for Ryzen 5600X (6c/12t)
BATCH_INSERT_SIZE = 64                 # number of rows per DB batch write
DB_QUEUE_MAX = 1000000000
TFIDF_EXPANSION = False                # disable by default for speed
DDGS_LIMIT_PER_QUERY = 6
REQUEST_TIMEOUT = 12
# ---------------------------

# Shared objects
session = requests.Session()
session.headers.update(USER_AGENT)

DB_LOCK = Lock()
doc_queue = queue.Queue(maxsize=DB_QUEUE_MAX)
log_queue = queue.Queue()
pages_counter_lock = Lock()
pages_crawled = 0
docs_saved = 0
stop_event = Event()

def log(msg):
    ts = time.strftime("%H:%M:%S")
    log_queue.put(f"[{ts}] {msg}")

# ---------------- Database ----------------
def init_db():
    conn = sqlite3.connect(DB, timeout=30)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_topic TEXT,
            source_url TEXT,
            text_content TEXT,
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    log("Database initialized (WAL mode)")

def db_writer_thread():
    global docs_saved
    conn = sqlite3.connect(DB, timeout=30)
    c = conn.cursor()
    buffer = []
    last_flush = time.time()
    while not (stop_event.is_set() and doc_queue.empty()):
        try:
            item = doc_queue.get(timeout=0.5)
            buffer.append(item)
            # flush if buffer full
            if len(buffer) >= BATCH_INSERT_SIZE:
                with DB_LOCK:
                    c.executemany(
                        "INSERT INTO documents (source_topic, source_url, text_content) VALUES (?, ?, ?)",
                        buffer
                    )
                    conn.commit()
                docs_saved += len(buffer)
                log(f"DB: inserted batch ({len(buffer)} rows). Total saved: {docs_saved}")
                buffer.clear()
            doc_queue.task_done()
        except queue.Empty:
            # flush periodically if we have items
            if buffer and (time.time() - last_flush) > 1.0:
                with DB_LOCK:
                    c.executemany(
                        "INSERT INTO documents (source_topic, source_url, text_content) VALUES (?, ?, ?)",
                        buffer
                    )
                    conn.commit()
                docs_saved += len(buffer)
                log(f"DB: flushed ({len(buffer)} rows). Total saved: {docs_saved}")
                buffer.clear()
                last_flush = time.time()
            continue
    # final flush
    if buffer:
        with DB_LOCK:
            c.executemany(
                "INSERT INTO documents (source_topic, source_url, text_content) VALUES (?, ?, ?)",
                buffer
            )
            conn.commit()
        docs_saved += len(buffer)
        log(f"DB: final flush ({len(buffer)} rows). Total saved: {docs_saved}")
    conn.close()
    log("DB writer thread exiting")

# ---------------- Utilities ----------------
def clean_text(text):
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join([l for l in lines if l])

def fetch_page(url, attempts=2):
    global pages_crawled
    if stop_event.is_set():
        return None
    try:
        r = session.get(url, timeout=REQUEST_TIMEOUT)
        if not r.ok:
            log(f"HTTP {r.status_code} for {url}")
            return None
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            tag.decompose()
        text = clean_text(soup.get_text())
        with pages_counter_lock:
            pages_crawled += 1
        return text
    except Exception as e:
        log(f"Fetch error {url}: {e}")
        return None

def is_url(s):
    return s.startswith("http://") or s.startswith("https://")

# ---------------- Search / Expansion ----------------
def crawl_query(query, limit=DDGS_LIMIT_PER_QUERY):
    urls = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                if stop_event.is_set():
                    break
                url = r.get("href")
                if url and is_url(url):
                    urls.append(url)
    except Exception as e:
        log(f"Search error for '{query}': {e}")
    return urls

def extract_keywords_fast(text, top_n=5):
    # lightweight: TF-IDF on concatenated text (only used if TFIDF_EXPANSION True)
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2000)
        tfidf = vec.fit_transform([text])
        scores = tfidf.toarray()[0]
        idx = scores.argsort()[-top_n:]
        return [vec.get_feature_names_out()[i] for i in idx if vec.get_feature_names_out()[i].isalpha()]
    except Exception as e:
        log(f"Keyword extraction error: {e}")
        return []

def expand_topic_with_web(topic):
    if stop_event.is_set():
        return []
    pages = crawl_query(topic, limit=3)
    collected = []
    for url in pages:
        if stop_event.is_set():
            break
        txt = fetch_page(url)
        if txt:
            collected.append(txt)
    if not collected or not TFIDF_EXPANSION:
        return []
    combined = " ".join(collected)
    kws = extract_keywords_fast(combined, top_n=8)
    log(f"Expanded '{topic}' -> {kws}")
    return kws

# ---------------- Worker ----------------
def crawl_topic(topic):
    if stop_event.is_set():
        return
    urls = [topic] if is_url(topic) else crawl_query(topic, limit=6)
    for url in urls:
        if stop_event.is_set():
            break
        text = fetch_page(url)
        if text:
            try:
                doc_queue.put_nowait((topic, url, text))
            except queue.Full:
                # if queue full, block minimally or drop
                try:
                    doc_queue.put((topic, url, text), timeout=2)
                except queue.Full:
                    log("DB queue full: dropping document")

# ---------------- Controller ----------------
def run_crawler_from_file():
    init_db()
    # start DB writer
    db_thread = threading.Thread(target=db_writer_thread, daemon=True)
    db_thread.start()

    # read inputs
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            base_topics = [line.strip() for line in f if line.strip()]
    except Exception as e:
        log(f"Failed to read input file: {e}")
        stop_event.set()
        return

    log(f"Loaded {len(base_topics)} base topics")
    # Expand topics (lightweight)
    all_topics = []
    for t in base_topics:
        if stop_event.is_set():
            break
        all_topics.append(t)
        kws = expand_topic_with_web(t)
        for k in kws:
            if k and not is_url(k):
                all_topics.append(k)
    # deduplicate
    all_topics = list(dict.fromkeys(all_topics))
    log(f"Total topics (after expansion): {len(all_topics)}")

    # Start thread pool
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(crawl_topic, t) for t in all_topics]
        try:
            for future in as_completed(futures):
                if stop_event.is_set():
                    break
                # handle exceptions
                try:
                    future.result()
                except Exception as e:
                    log(f"Worker exception: {e}")
        except KeyboardInterrupt:
            log("KeyboardInterrupt received, stopping")
            stop_event.set()

    log("Workers complete, waiting for DB queue to drain")
    # wait for queue drained
    doc_queue.join()
    # signal DB writer to finish via stop_event
    stop_event.set()
    db_thread.join(timeout=10)
    log("Crawler finished")

# ---------------- GUI ----------------
class CrawlerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Optimized Dataset Crawler")
        root.geometry("1000x700")

        top_frame = ttk.Frame(root)
        top_frame.pack(fill=tk.X, padx=8, pady=6)

        self.start_btn = ttk.Button(top_frame, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(top_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(top_frame, text="Max threads:").pack(side=tk.LEFT, padx=(12, 0))
        self.threads_var = tk.IntVar(value=MAX_THREADS)
        self.threads_entry = ttk.Entry(top_frame, width=6, textvariable=self.threads_var)
        self.threads_entry.pack(side=tk.LEFT)

        ttk.Label(top_frame, text="TF-IDF expand:").pack(side=tk.LEFT, padx=(12, 0))
        self.tfidf_var = tk.BooleanVar(value=TFIDF_EXPANSION)
        self.tfidf_cb = ttk.Checkbutton(top_frame, variable=self.tfidf_var)
        self.tfidf_cb.pack(side=tk.LEFT)

        metrics_frame = ttk.Frame(root)
        metrics_frame.pack(fill=tk.X, padx=8)

        self.pages_label = ttk.Label(metrics_frame, text="Pages crawled: 0")
        self.pages_label.pack(side=tk.LEFT, padx=(0, 12))
        self.docs_label = ttk.Label(metrics_frame, text="Docs saved: 0")
        self.docs_label.pack(side=tk.LEFT, padx=(0, 12))
        self.rate_label = ttk.Label(metrics_frame, text="Pages/sec: 0.00")
        self.rate_label.pack(side=tk.LEFT)

        self.log_box = ScrolledText(root, state='normal', height=30, font=("Consolas", 10))
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        # log processing timer
        self.root.after(100, self.process_log)
        # metrics update
        self._last_pages = 0
        self._last_time = time.time()
        self.root.after(1000, self.update_metrics)

        self.worker_thread = None

    def process_log(self):
        updated = False
        try:
            while True:
                msg = log_queue.get_nowait()
                self.log_box.insert(tk.END, msg + "\n")
                self.log_box.see(tk.END)
                updated = True
        except queue.Empty:
            pass
        self.root.after(100, self.process_log)

    def update_metrics(self):
        global pages_crawled, docs_saved
        now = time.time()
        pages = pages_crawled
        delta_pages = pages - self._last_pages
        delta_t = now - self._last_time
        rate = (delta_pages / delta_t) if delta_t > 0 else 0.0
        self.pages_label.config(text=f"Pages crawled: {pages}")
        self.docs_label.config(text=f"Docs saved: {docs_saved}")
        self.rate_label.config(text=f"Pages/sec: {rate:.2f}")
        self._last_pages = pages
        self._last_time = now
        self.root.after(1000, self.update_metrics)

    def start(self):
        global MAX_THREADS, TFIDF_EXPANSION
        if self.worker_thread and self.worker_thread.is_alive():
            log("Already running")
            return
        # update config from UI
        try:
            MAX_THREADS = max(4, int(self.threads_var.get()))
        except Exception:
            MAX_THREADS = 180
        TFIDF_EXPANSION = bool(self.tfidf_var.get())
        # reset counters and flags
        stop_event.clear()
        self.worker_thread = threading.Thread(target=run_crawler_from_file, daemon=True)
        self.worker_thread.start()
        log("Crawler started")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        if not stop_event.is_set():
            log("Stop requested")
            stop_event.set()
        self.stop_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.NORMAL)

def main():
    root = tk.Tk()
    app = CrawlerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
