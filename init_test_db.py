import sqlite3

conn = sqlite3.connect("test_dq.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    job TEXT,
    balance REAL
)
""")

cur.execute("DELETE FROM customers")

rows = [
    (1, "Amit", 30, "admin.", 1000),
    (2, "Neha", 150, "services", -200),   # bad age, negative balance
    (3, "Raj", None, None, 500),          # missing age, job
]
cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?)", rows)

conn.commit()
conn.close()
print("test_dq.db initialized.")
