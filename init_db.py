import sqlite3
import os

os.makedirs("database", exist_ok=True)

conn   = sqlite3.connect("database/buildings.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS buildings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    building_name TEXT UNIQUE,
    condition     TEXT DEFAULT 'Good',
    risk_score    INTEGER DEFAULT 0,
    image_path    TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    building_name   TEXT,
    condition       TEXT,
    confidence      REAL,
    risk_score      INTEGER,
    assessment_date TEXT,
    image_path      TEXT
)
""")

conn.commit()
conn.close()
print("Database initialised successfully. Tables created: buildings, assessments")