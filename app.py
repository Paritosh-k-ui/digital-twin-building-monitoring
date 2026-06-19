from flask import Flask, render_template, request
from detector import predict_condition, extract_image_metrics, compute_combined_score
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
# ASSESS BUILDING
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

    # ── ML classification ────────────────────────────────────────────────
    result     = predict_condition(filepath)
    condition  = result["condition"]
    confidence = result["confidence"]

    # ── Image-derived CV metrics (from actual pixels) ────────────────────
    img_metrics = extract_image_metrics(filepath, ml_condition=condition, ml_confidence=confidence)

    # ── Combined weighted score (environmental metrics removed) ──────────
    scores          = compute_combined_score(condition, confidence, img_metrics)
    combined_score  = scores["combined_score"]
    final_condition = scores["final_condition"]
    ml_score        = scores["ml_score"]
    cv_score        = scores["cv_score"]

    assessment_date = datetime.now().strftime("%d-%m-%Y %H:%M")

    # ── Persist ──────────────────────────────────────────────────────────
    try:
        conn   = sqlite3.connect("database/buildings.db")
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO buildings
            (building_name, condition, risk_score, image_path,
             crack_density, discolouration, tilt, vegetation, surface_roughness,
             ml_score, cv_score, env_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            building_name, final_condition, combined_score, filepath,
            img_metrics["crack_density"], img_metrics["discolouration"],
            img_metrics["tilt"],          img_metrics["vegetation"],
            img_metrics["surface_roughness"],
            ml_score, cv_score
        ))

        cursor.execute("""
            INSERT INTO assessments
            (building_name, condition, confidence, risk_score, assessment_date, image_path,
             crack_density, discolouration, tilt, vegetation, surface_roughness,
             ml_score, cv_score, env_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            building_name, final_condition, round(confidence * 100, 1),
            combined_score, assessment_date, filepath,
            img_metrics["crack_density"], img_metrics["discolouration"],
            img_metrics["tilt"],          img_metrics["vegetation"],
            img_metrics["surface_roughness"],
            ml_score, cv_score
        ))

        conn.commit()
        conn.close()
    except Exception as e:
        print("DB Error:", e)

    # ── Render result using result.html template ─────────────────────────
    conf_pct     = round(confidence * 100, 1)
    border_color = {"Critical": "#ef4444", "Warning": "#f59e0b", "Good": "#22c55e"}.get(final_condition, "#38bdf8")
    icon         = {"Critical": "🔴", "Warning": "🟡", "Good": "🟢"}.get(final_condition, "⚪")
    rec          = {
        "Critical": "Immediate structural intervention required. Building poses safety risk. Evacuate and schedule emergency repairs.",
        "Warning":  "Moderate risk detected. Schedule professional structural inspection within 30 days and plan maintenance.",
        "Good":     "Building is in good structural condition. Continue regular maintenance schedule and annual inspections.",
    }.get(final_condition, "")

    return render_template(
        "result.html",
        building_name=building_name,
        filepath=filepath,
        final_condition=final_condition,
        combined_score=combined_score,
        ml_score=ml_score,
        cv_score=cv_score,
        conf_pct=conf_pct,
        assessment_date=assessment_date,
        img_metrics=img_metrics,
        border_color=border_color,
        icon=icon,
        rec=rec
    )


# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():
    conn   = sqlite3.connect("database/buildings.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT building_name, condition, risk_score, image_path,
               crack_density, discolouration, tilt, vegetation, surface_roughness,
               ml_score, cv_score, env_score
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


# =========================
# DB SEEDER
# =========================

def seed_database_if_empty():
    import shutil
    conn = sqlite3.connect("database/buildings.db")
    cursor = conn.cursor()
    
    # Check if there are any buildings
    cursor.execute("SELECT COUNT(*) FROM buildings")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("[SEED] Seeding database with initial sample data...")
        seeds = [
            ("Block A - Central Library", "dataset/good/1.jpg", "seed_good.jpg"),
            ("Block B - South Annex", "dataset/warning/1.jpg", "seed_warning.jpg"),
            ("Block C - Parking Garage", "dataset/critical/1 (1).jpg", "seed_critical.jpg")
        ]
        
        for name, src, dest_filename in seeds:
            if os.path.exists(src):
                dest_path = os.path.join(app.config["UPLOAD_FOLDER"], dest_filename)
                shutil.copy(src, dest_path)
                
                # Perform analysis
                result = predict_condition(dest_path)
                metrics = extract_image_metrics(dest_path, ml_condition=result["condition"], ml_confidence=result["confidence"])
                scores = compute_combined_score(result["condition"], result["confidence"], metrics)
                
                combined_score = scores["combined_score"]
                final_condition = scores["final_condition"]
                ml_score = scores["ml_score"]
                cv_score = scores["cv_score"]
                
                # Insert into DB
                cursor.execute("""
                    INSERT OR REPLACE INTO buildings
                    (building_name, condition, risk_score, image_path,
                     crack_density, discolouration, tilt, vegetation, surface_roughness,
                     ml_score, cv_score, env_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    name, final_condition, combined_score, dest_path,
                    metrics["crack_density"], metrics["discolouration"],
                    metrics["tilt"], metrics["vegetation"], metrics["surface_roughness"],
                    ml_score, cv_score
                ))
                
                # Insert into assessments
                assessment_date = datetime.now().strftime("%d-%m-%Y %H:%M")
                cursor.execute("""
                    INSERT INTO assessments
                    (building_name, condition, confidence, risk_score, assessment_date, image_path,
                     crack_density, discolouration, tilt, vegetation, surface_roughness,
                     ml_score, cv_score, env_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    name, final_condition, round(result["confidence"] * 100, 1),
                    combined_score, assessment_date, dest_path,
                    metrics["crack_density"], metrics["discolouration"],
                    metrics["tilt"], metrics["vegetation"], metrics["surface_roughness"],
                    ml_score, cv_score
                ))
        conn.commit()
        print("[SEED] Seeding completed.")
    conn.close()


if __name__ == "__main__":
    seed_database_if_empty()
    app.run(debug=True)