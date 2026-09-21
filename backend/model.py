import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

DATA_FILE = "../data/flood_data.csv"
MODEL_FILE = "../model/flood_model.pkl"

data = pd.read_csv(DATA_FILE)

X = data[
    [
        "rainfall",
        "water_level",
        "river_level",
        "soil_moisture",
        "humidity"
    ]
]

y = data["flood_risk"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

os.makedirs("../model", exist_ok=True)

joblib.dump(model, MODEL_FILE)

print("AI flood prediction model trained successfully.")
