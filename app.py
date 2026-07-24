import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load trained Decision Tree model
MODEL_PATH = "DecisionTree.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print(f"Warning: '{MODEL_PATH}' not found. Please place it in the project root.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model file not found on server."}), 500

    try:
        # Extract features from form input
        age = float(request.form.get("Age"))
        gender = float(request.form.get("Gender"))
        region = float(request.form.get("Region"))
        occupation = float(request.form.get("Occupation"))
        income = float(request.form.get("Income"))

        # Combine features in expected sequence: Age, Gender, Region, Occupation, Income
        features = np.array([[age, gender, region, occupation, income]])

        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability if model supports it
        probabilities = model.predict_proba(features)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)

        return jsonify({
            "status": "success",
            "prediction": str(prediction).upper(),
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
