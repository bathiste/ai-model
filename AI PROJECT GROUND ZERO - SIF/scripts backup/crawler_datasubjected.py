import requests
import sqlite3
import time
import nltk
from bs4 import BeautifulSoup
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

nltk.download('punkt', quiet=True)

DB = "dataset.db"
INPUT_FILE = "crawler_topicinputs.txt"
USER_AGENT = {"User-Agent": "Mozilla/5.0 (DatasetBuilder/3.0-threaded)"}

MAX_THREADS = 1000 # Increase to 20–30 depending on your hardware
DB_LOCK = Lock()  # Protect SQLite

# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
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

def save_document(topic, url, text):
    with DB_LOCK:
        try:
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            c.execute(
                "INSERT INTO documents (source_topic, source_url, text_content) VALUES (?, ?, ?)",
                (topic, url, text)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print("DB error:", e)

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def clean_text(text):
    if not text:
        return ""
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join([l for l in lines if l])

def fetch_page(url, attempts=2):
    for _ in range(attempts):
        try:
            r = requests.get(url, headers=USER_AGENT, timeout=10)
            if r.ok:
                soup = BeautifulSoup(r.text, "html.parser")
                for e in soup(["script", "style", "header", "footer", "nav"]):
                    e.decompose()
                return clean_text(soup.get_text())
        except:
            time.sleep(1)
    return None

def is_url(s):
    return s.startswith("http://") or s.startswith("https://")

# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------
def crawl_query(query, limit=5):
    urls = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                url = r.get("href")
                if url and is_url(url):
                    urls.append(url)
    except:
        pass
    return urls

# ---------------------------------------------------------
# TOPIC EXPANSION
# ---------------------------------------------------------
def extract_keywords(text, top_n=5):
    if not text or len(text) < 20:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform([text])
        scores = tfidf.toarray()[0]
        idx = scores.argsort()[-top_n:]
        return [vec.get_feature_names_out()[i] for i in idx]
    except:
        return []

def expand_topic_with_web(topic):
    pages = crawl_query(topic, limit=3)
    collected = []

    for url in pages:
        txt = fetch_page(url)
        if txt:
            collected.append(txt)
        time.sleep(0.5)

    if not collected:
        return []

    combined = " ".join(collected)
    kws = extract_keywords(combined, top_n=10)

    return [k for k in kws if len(k) > 3 and k.isalpha()]

# ---------------------------------------------------------
# PARALLEL CRAWLING WORKER
# ---------------------------------------------------------
def crawl_topic(topic):
    urls = [topic] if is_url(topic) else crawl_query(topic, limit=5)

    for url in urls:
        text = fetch_page(url)
        if text:
            save_document(topic, url, text)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run():
    init_db()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        base_topics = [line.strip() for line in f if line.strip()]

    print("Expanding topics...")
    expanded = []

    for topic in base_topics:
        expanded.append(topic)
        kws = expand_topic_with_web(topic)
        expanded.extend(kws)

    all_topics = list(set(expanded))
    print("Total topics:", len(all_topics))

    print("Crawling...")
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(crawl_topic, t) for t in all_topics]

        for f in as_completed(futures):
            pass  # Just ensuring tasks complete

    print("DONE.")

if __name__ == "__main__":
    run()
