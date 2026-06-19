import sqlite3
import os

os.makedirs("database", exist_ok=True)

conn   = sqlite3.connect("database/buildings.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS buildings (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    building_name      TEXT UNIQUE,
    condition          TEXT DEFAULT 'Good',
    risk_score         INTEGER DEFAULT 0,
    image_path         TEXT,
    crack_density      INTEGER DEFAULT 0,
    discolouration     INTEGER DEFAULT 0,
    tilt               INTEGER DEFAULT 0,
    vegetation         INTEGER DEFAULT 0,
    surface_roughness  INTEGER DEFAULT 0,
    ml_score           INTEGER DEFAULT 0,
    cv_score           INTEGER DEFAULT 0,
    env_score          INTEGER DEFAULT 0
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS assessments (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    building_name      TEXT,
    condition          TEXT,
    confidence         REAL,
    risk_score         INTEGER,
    assessment_date    TEXT,
    image_path         TEXT,
    crack_density      INTEGER DEFAULT 0,
    discolouration     INTEGER DEFAULT 0,
    tilt               INTEGER DEFAULT 0,
    vegetation         INTEGER DEFAULT 0,
    surface_roughness  INTEGER DEFAULT 0,
    ml_score           INTEGER DEFAULT 0,
    cv_score           INTEGER DEFAULT 0,
    env_score          INTEGER DEFAULT 0
)
""")

# ── Migrate existing DB: add new columns if they don't exist yet ──────────────
new_cols = [
    ("crack_density",     "INTEGER DEFAULT 0"),
    ("discolouration",    "INTEGER DEFAULT 0"),
    ("tilt",              "INTEGER DEFAULT 0"),
    ("vegetation",        "INTEGER DEFAULT 0"),
    ("surface_roughness", "INTEGER DEFAULT 0"),
    ("ml_score",          "INTEGER DEFAULT 0"),
    ("cv_score",          "INTEGER DEFAULT 0"),
    ("env_score",         "INTEGER DEFAULT 0"),
]
for table in ("buildings", "assessments"):
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    for col, typedef in new_cols:
        if col not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
            print(f"  Migrated: added {col} to {table}")

conn.commit()
conn.close()
print("Database initialised successfully.")