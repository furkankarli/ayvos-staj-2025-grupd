import sqlite3

DB_PATH = "plates.db"


def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS plates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT NOT NULL,
        filename TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    )
    conn.commit()
    conn.close()


def insert_plate(plate, filename):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO plates (plate, filename) VALUES (?, ?)", (plate, filename)
    )
    conn.commit()
    conn.close()


def fetch_plates():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT plate, filename, timestamp FROM plates ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


# Tabloyu oluştur
create_table()
