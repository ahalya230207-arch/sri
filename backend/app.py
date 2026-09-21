from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "flood_model.pkl"
)

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


@app.route("/")
def home():

    return jsonify({
        "system": "AI Flood Risk Detection & Early Warning System",
        "status": "Backend Online"
    })


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        return jsonify({
            "error": "AI model not found. Train the model first."
        }), 500

    data = request.get_json()

    rainfall = float(data["rainfall"])
    water_level = float(data["water_level"])
    river_level = float(data["river_level"])
    soil_moisture = float(data["soil_moisture"])
    humidity = float(data["humidity"])

    prediction = model.predict([[
        rainfall,
        water_level,
        river_level,
        soil_moisture,
        humidity
    ]])

    risk_score = round(float(prediction[0]), 2)

    risk_score = max(0, min(100, risk_score))

    if risk_score < 30:

        risk_level = "LOW"
        warning = "Normal monitoring recommended."

    elif risk_score < 60:

        risk_level = "MODERATE"
        warning = "Monitor rainfall and water levels."

    elif risk_score < 80:

        risk_level = "HIGH"
        warning = "Flood warning. Prepare emergency response."

    else:

        risk_level = "CRITICAL"
        warning = "Critical flood risk. Immediate emergency preparedness required."

    return jsonify({
        "risk_score": risk_score,
        "risk_level": risk_level,
        "warning": warning
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
