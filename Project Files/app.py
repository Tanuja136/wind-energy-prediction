from flask import Flask, render_template, request
import joblib
import numpy as np
import requests

app = Flask(__name__)

# Load trained ML model and scaler
model = joblib.load("wind_power_model.pkl")
scaler = joblib.load("scaler.pkl")

# 🔑 Your OpenWeather API key
API_KEY = "feef6773a4dec2a99a50c5ae96d75c37"

# Dropdown cities
CITIES = ["Agartala", "London", "New York", "Delhi", "Mumbai", "Chennai"]

@app.route("/")
def home():
    return render_template("intro.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    weather_data = None
    prediction = None

    if request.method == "POST":

        # ---------- WEATHER SECTION ----------
        if "city" in request.form:
            city = request.form.get("city")

            if city:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

                try:
                    response = requests.get(url)
                    data = response.json()

                    if response.status_code == 200 and "main" in data:
                        weather_data = {
                            "city": city,
                            "temperature": round(data["main"]["temp"], 2),
                            "humidity": data["main"]["humidity"],
                            "pressure": data["main"]["pressure"],
                            "wind_speed": data["wind"]["speed"]
                        }
                    else:
                        weather_data = {
                            "city": city,
                            "temperature": "Error",
                            "humidity": "-",
                            "pressure": "-",
                            "wind_speed": "-"
                        }

                except Exception as e:
                    print("Weather API Error:", e)
                    weather_data = {
                        "city": city,
                        "temperature": "API Error",
                        "humidity": "-",
                        "pressure": "-",
                        "wind_speed": "-"
                    }

        # ---------- PREDICTION SECTION ----------
        if "theoretical_power" in request.form and "wind_speed" in request.form:
            try:
                theoretical_power = float(request.form.get("theoretical_power"))
                wind_speed = float(request.form.get("wind_speed"))

                # IMPORTANT: Order must match training: [Wind Speed, Theoretical Power]
                features = np.array([[wind_speed, theoretical_power]])
                features_scaled = scaler.transform(features)

                result = model.predict(features_scaled)[0]
                prediction = round(result, 2)

            except Exception as e:
                print("Prediction Error:", e)
                prediction = "Invalid Input"

    return render_template(
        "predict.html",
        cities=CITIES,
        weather_data=weather_data,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
