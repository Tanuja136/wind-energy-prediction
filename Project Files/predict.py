import joblib
import numpy as np

# Load model and scaler
model = joblib.load("wind_power_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example input: [Wind Speed, Theoretical Power]
wind_speed = 8.5
theoretical_power = 900

features = np.array([[wind_speed, theoretical_power]])
features_scaled = scaler.transform(features)

prediction = model.predict(features_scaled)

print("Predicted Power Output:", prediction[0])
