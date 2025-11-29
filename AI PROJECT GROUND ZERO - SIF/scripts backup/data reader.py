import sqlite3

conn = sqlite3.connect("dataset.db")
columns = conn.execute("PRAGMA table_info(documents);").fetchall()
conn.close()

print("Columns:", columns)

