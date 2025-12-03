# services/db_service.py
import sqlite3
import pandas as pd

DB_PATH = "test_dq.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

def read_table(table_name: str, limit: int | None = None) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with get_conn() as conn:
        return pd.read_sql_query(query, conn)

def execute_sql(query: str) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(query, conn)
