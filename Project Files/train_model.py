import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load dataset (CSV)
data = pd.read_csv("data/T1.csv")

# Check column names once if needed
# print(data.columns)

# Select features and target
X = data[["Wind Speed (m/s)", "Theoretical_Power_Curve (KWh)"]]
y = data["LV ActivePower (kW)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R2 Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# Plot: Actual vs Predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Power")
plt.ylabel("Predicted Power")
plt.title("Actual vs Predicted Wind Power Output")
plt.grid(True)
plt.savefig("training_result.png")
plt.close()

# Save model and scaler
joblib.dump(model, "wind_power_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved as wind_power_model.pkl")
print("Scaler saved as scaler.pkl")
print("Training plot saved as training_result.png")
