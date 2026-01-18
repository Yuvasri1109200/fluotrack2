from flask import Flask, request, jsonify, send_from_directory, session, render_template
from db import get_connection
from ultralytics import YOLO
import cv2
import os
import uuid

yolo_model = YOLO("yolov8n.pt")  # your trained model
app = Flask(__name__, template_folder='templates')
app.secret_key = 'supersecret'

@app.route("/api/yolo_detect", methods=["POST"])
def yolo_detect():
    try:
        # Get uploaded image
        file = request.files["image"]
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join("static", "uploads", filename)
        os.makedirs("static/uploads", exist_ok=True)
        file.save(filepath)

        # Run YOLO detection
        results = yolo_model.predict(filepath)[0]

        # Count detections
        particle_count = len(results.boxes)

        # Draw boxes on image
        annotated_path = os.path.join("static", "outputs", filename)
        os.makedirs("static/outputs", exist_ok=True)
        results.save(filename=annotated_path)

        return jsonify({
            "count": particle_count,
            "image_url": f"/static/outputs/{filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Get uploaded image
        file = request.files["image"]
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join("static", "uploads", filename)
        os.makedirs("static/uploads", exist_ok=True)
        file.save(filepath)

        # Run YOLO prediction
        results = yolo_model.predict(filepath)[0]

        # Get the highest confidence
        if len(results.boxes) > 0:
            max_conf = max(results.boxes.conf.cpu().numpy())
            result = "microplastic"
            probability = float(max_conf)
        else:
            result = "no detection"
            probability = 0.0

        return jsonify({
            "result": result,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------ USER AUTH ------------------

@app.route("/api/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not all([name, email, password]):
            return jsonify({"error": "Missing fields"}), 400

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        if user:
            return jsonify({"error": "Email already exists"}), 400

        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, password)
        )
        conn.commit()
        return jsonify({"message": "User registered successfully"}), 201

    except Exception as e:
        return jsonify({"error": f"Signup failed: {str(e)}"}), 500


@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not all([email, password]):
            return jsonify({"error": "Missing fields"}), 400

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        )
        user = cursor.fetchone()

        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        # Set session variables
        session['user_id'] = user[0]
        session['user_name'] = user[1]

        return jsonify({"id": user[0], "name": user[1], "email": user[2]})

    except Exception as e:
        return jsonify({"error": f"Login failed: {str(e)}"}), 500


@app.route("/api/current_user")
def current_user():
    try:
        if 'user_id' in session:
            return jsonify({
                "id": session['user_id'],
                "name": session.get('user_name', '')
            })
        return jsonify(None)
    except Exception as e:
        return jsonify({"error": f"Current user check failed: {str(e)}"}), 500


# ------------------ SAMPLES ------------------

@app.route("/api/samples", methods=["POST"])
def save_sample():
    try:
        data = request.get_json()
        required_fields = ["user_id", "sample_id", "location", "operator", "particle_count"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing sample fields"}), 400

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO samples (user_id, sample_id, location, operator, particle_count) 
               VALUES (?, ?, ?, ?, ?)""",
            (data["user_id"], data["sample_id"], data["location"], data["operator"], data["particle_count"])
        )
        conn.commit()
        return jsonify({"message": "Sample saved"}), 201
    except Exception as e:
        return jsonify({"error": f"Save sample failed: {str(e)}"}), 500


@app.route("/api/samples/<int:user_id>", methods=["GET"])
def get_samples(user_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM samples WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
        samples = cursor.fetchall()
        # Convert to list of dicts
        sample_list = []
        for s in samples:
            sample_list.append({
                "id": s[0],
                "user_id": s[1],
                "sample_id": s[2],
                "location": s[3],
                "operator": s[4],
                "particle_count": s[5],
                "timestamp": s[6]
            })
        return jsonify(sample_list)
    except Exception as e:
        return jsonify({"error": f"Get samples failed: {str(e)}"}), 500


@app.route("/api/samples/delete/<int:sample_id>", methods=["DELETE"])
def delete_sample(sample_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM samples WHERE id=?", (sample_id,))
        conn.commit()
        return jsonify({"message": "Sample deleted"})
    except Exception as e:
        return jsonify({"error": f"Delete sample failed: {str(e)}"}), 500


# ------------------ FRONTEND ------------------

@app.route("/")
def index():
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True, port=5500)
