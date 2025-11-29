import sqlite3

DB = "dataset.db"
OUTFILE = "train.txt"

def export_text():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT text_content FROM documents")

    rows = c.fetchall()
    conn.close()

    with open(OUTFILE, "w", encoding="utf-8") as f:
        for row in rows:
            text = (row[0] or "").strip()
            if len(text) > 0:
                # Wrap each entry in instruction format (good for LLMs)
                f.write("### Instruction:\n")
                f.write("Summarize the following document.\n\n")
                f.write("### Document:\n")
                f.write(text + "\n\n")
                f.write("### Response:\n\n")
                f.write("### End\n\n")

    print(f"Export complete. Wrote {OUTFILE}")

if __name__ == "__main__":
    export_text()
