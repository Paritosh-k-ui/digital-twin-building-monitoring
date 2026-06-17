import sqlite3
import os

db_path = "database/buildings.db"

if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Deleted old database: {db_path}")
else:
    print("No existing database found.")

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE buildings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    building_name TEXT UNIQUE,
    condition     TEXT DEFAULT 'Good',
    risk_score    INTEGER DEFAULT 0,
    image_path    TEXT
)
""")

cursor.execute("""
CREATE TABLE assessments (
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

conn = sqlite3.connect(db_path)
print("\nbuildings table columns:")
print(conn.execute("PRAGMA table_info(buildings)").fetchall())
print("\nassessments table columns:")
print(conn.execute("PRAGMA table_info(assessments)").fetchall())
conn.close()

print("\nDatabase reset complete with simplified schema.")