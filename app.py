from flask import Flask, render_template, request
from detector import predict_condition
from datetime import datetime
import sqlite3
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# ASSESS BUILDING (Image only)
# =========================

@app.route("/assess", methods=["POST"])
def assess():
    building_name = request.form.get("building_name", "Unknown Building")

    if "image" not in request.files:
        return "No image selected"

    file = request.files["image"]
    if file.filename == "":
        return "No image selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # ── ML Image Classification ─────────────────────────────────────────────
    result     = predict_condition(filepath)
    condition  = result["condition"]    # Good / Warning / Critical
    confidence = result["confidence"]
    color      = result["color"]
    # ─────────────────────────────────────────────────────────────────────────

    risk_score = {"Good": 15, "Warning": 50, "Critical": 85}.get(condition, 50)

    assessment_date = datetime.now().strftime("%d-%m-%Y %H:%M")

    try:
        conn   = sqlite3.connect("database/buildings.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO buildings
            (building_name, condition, risk_score, image_path)
            VALUES (?, ?, ?, ?)
        """, (building_name, condition, risk_score, filepath))

        cursor.execute("""
            INSERT INTO assessments
            (building_name, condition, confidence, risk_score, assessment_date, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (building_name, condition, round(confidence * 100, 1), risk_score, assessment_date, filepath))

        conn.commit()
        conn.close()
    except Exception as e:
        print("DB Error:", e)

    conf_pct     = round(confidence * 100, 1)
    border_color = {"Critical": "#ef4444", "Warning": "#f59e0b", "Good": "#22c55e"}.get(condition, "#38bdf8")
    icon         = {"Critical": "🔴", "Warning": "🟡", "Good": "🟢"}.get(condition, "⚪")

    return f"""
<!DOCTYPE html>
<html>
<head>
<title>Assessment Result</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; font-family:'Segoe UI',Arial,sans-serif; }}
body {{ background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%); min-height:100vh; display:flex; align-items:center; justify-content:center; padding:30px; }}
.card {{ background:rgba(30,41,59,0.95); backdrop-filter:blur(10px); border-radius:24px; padding:36px; width:100%; max-width:700px; border:2px solid {border_color}; box-shadow:0 0 60px rgba(0,0,0,0.5),0 0 30px {border_color}33; }}
h1 {{ font-size:1.7rem; color:white; margin-bottom:20px; text-align:center; }}
.img-wrap {{ text-align:center; margin-bottom:20px; }}
.img-wrap img {{ max-width:100%; max-height:300px; border-radius:14px; border:1px solid #334155; }}
.condition-box {{ background:#0f172a; border-radius:16px; padding:24px; text-align:center; margin-bottom:20px; border:1px solid {border_color}; }}
.condition-icon {{ font-size:2.8rem; margin-bottom:6px; }}
.condition-label {{ font-size:1.9rem; font-weight:bold; color:{border_color}; }}
.conf-text {{ color:#94a3b8; font-size:0.9rem; margin-top:6px; }}
.risk-wrap {{ margin:18px 0; }}
.risk-label {{ display:flex; justify-content:space-between; font-size:0.85rem; color:#94a3b8; margin-bottom:6px; }}
.risk-bar {{ height:12px; background:#1e293b; border-radius:6px; overflow:hidden; }}
.risk-fill {{ height:100%; border-radius:6px; background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444); width:{risk_score}%; }}
.details {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:20px; }}
.detail-item {{ background:#0f172a; border-radius:12px; padding:14px; border:1px solid #334155; }}
.detail-label {{ font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px; }}
.detail-val {{ font-size:0.95rem; color:white; font-weight:600; }}
.rec-box {{ background:#0f172a; border-radius:12px; padding:16px; border-left:4px solid {border_color}; margin-bottom:20px; }}
.rec-box h3 {{ color:{border_color}; font-size:0.85rem; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em; }}
.rec-box p {{ color:#cbd5e1; font-size:0.88rem; line-height:1.6; }}
.btns {{ display:flex; gap:10px; flex-wrap:wrap; }}
.btn {{ flex:1; text-align:center; padding:12px; border-radius:12px; text-decoration:none; font-weight:bold; font-size:0.88rem; transition:opacity 0.2s; min-width:140px; }}
.btn:hover {{ opacity:0.85; }}
.btn-blue  {{ background:linear-gradient(135deg,#38bdf8,#0ea5e9); color:black; }}
.btn-green {{ background:#134e22; color:#22c55e; border:1px solid #22c55e; }}
.btn-amber {{ background:#451a03; color:#f59e0b; border:1px solid #f59e0b; }}
</style>
</head>
<body>
<div class="card">
    <h1>🏢 Building Assessment Result</h1>

    <div class="img-wrap">
        <img src="/{filepath}" alt="Building image">
    </div>

    <div class="condition-box">
        <div class="condition-icon">{icon}</div>
        <div class="condition-label">{condition}</div>
        <div class="conf-text">ML Confidence: {conf_pct}%</div>
    </div>

    <div class="risk-wrap">
        <div class="risk-label"><span>Estimated Risk Level</span><span>{risk_score}/100</span></div>
        <div class="risk-bar"><div class="risk-fill"></div></div>
    </div>

    <div class="details">
        <div class="detail-item"><div class="detail-label">Building</div><div class="detail-val">{building_name}</div></div>
        <div class="detail-item"><div class="detail-label">Assessed On</div><div class="detail-val">{assessment_date}</div></div>
    </div>

    <div class="rec-box">
        <h3>🔧 Recommendation</h3>
        <p>
        {"Immediate structural intervention required. Building poses safety risk. Evacuate and schedule emergency repairs." if condition == "Critical"
        else "Moderate risk detected. Schedule professional structural inspection within 30 days and plan maintenance." if condition == "Warning"
        else "Building is in good structural condition. Continue regular maintenance schedule and annual inspections."}
        </p>
    </div>

    <div class="btns">
        <a href="/" class="btn btn-blue">+ New Assessment</a>
        <a href="/dashboard" class="btn btn-green">🏙 Digital Twin</a>
        <a href="/history" class="btn btn-amber">📋 History</a>
    </div>
</div>
</body>
</html>
"""


# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():
    conn   = sqlite3.connect("database/buildings.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT building_name, condition, risk_score, image_path
        FROM buildings
    """)
    buildings = cursor.fetchall()

    cursor.execute("""
        SELECT building_name, condition, risk_score, assessment_date
        FROM assessments ORDER BY id DESC LIMIT 1
    """)
    latest = cursor.fetchone()

    conn.close()
    return render_template("dashboard.html", buildings=buildings, latest=latest)


# =========================
# HISTORY
# =========================

@app.route("/history")
def history():
    conn   = sqlite3.connect("database/buildings.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT building_name, condition, confidence, risk_score, assessment_date
        FROM assessments ORDER BY id DESC
    """)
    records = cursor.fetchall()

    conn.close()
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)