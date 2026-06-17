import sqlite3

conn   = sqlite3.connect("database/buildings.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM assessments")
cursor.execute("DELETE FROM buildings")

conn.commit()
conn.close()
print("Database reset. All buildings and assessments cleared.")