

"""
Iteration 3 Refined: Weather Prediction with Usability, Validation, and Model Accuracy
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import datetime

# --------------------------
# Generate synthetic hourly weather data (temperature in Celsius, visibility in km, UV index)
np.random.seed(42)
hours = 24
base_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
timestamps = [base_time + datetime.timedelta(hours=i) for i in range(hours)]

# True values (simulate a day)
true_temperature = 15 + 10 * np.sin(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 1, hours)
true_visibility = 8 + 2 * np.cos(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 0.5, hours)
true_uv_index = np.clip(6 * np.maximum(0, np.sin(np.linspace(-np.pi/2, 3*np.pi/2, hours))), 0, 11) + np.random.normal(0, 0.3, hours)

# --------------------------
# Data Validation & Optional Filtering
def validate_and_filter(temp, vis, uv):
    # Define reasonable ranges
    temp_valid = np.clip(temp, -30, 50)  # Celsius
    vis_valid = np.clip(vis, 0, 20)      # km
    uv_valid = np.clip(uv, 0, 11)        # UV index scale
    return temp_valid, vis_valid, uv_valid

temperature, visibility, uv_index = validate_and_filter(true_temperature, true_visibility, true_uv_index)

# --------------------------
# Simple prediction model: previous value + noise (for demonstration)
def predict_series(series):
    pred = np.zeros_like(series)
    pred[0] = series[0]
    for i in range(1, len(series)):
        pred[i] = series[i-1] + np.random.normal(0, 0.5)
    return pred

pred_temperature = predict_series(temperature)
pred_visibility = predict_series(visibility)
pred_uv_index = predict_series(uv_index)

# --------------------------
# Model Accuracy: Mean Squared Error
mse_temp = mean_squared_error(temperature, pred_temperature)
mse_vis = mean_squared_error(visibility, pred_visibility)
mse_uv = mean_squared_error(uv_index, pred_uv_index)

print(f"Mean Squared Error (Temperature): {mse_temp:.2f} °C^2")
print(f"Mean Squared Error (Visibility): {mse_vis:.2f} km^2")
print(f"Mean Squared Error (UV Index): {mse_uv:.2f}")

# --------------------------
# Save predictions to file with units and hourly timestamps
with open("iteration3_predictions.txt", "w") as f:
    f.write("Hour\tTimestamp\tTemperature (°C)\tVisibility (km)\tUV Index\n")
    for i in range(hours):
        f.write(f"{i+1}\t{timestamps[i].strftime('%Y-%m-%d %H:%M')}\t"
                f"{pred_temperature[i]:.2f}\t{pred_visibility[i]:.2f}\t{pred_uv_index[i]:.2f}\n")

# --------------------------
# Plot results with units and hourly ticks
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
hour_labels = [t.strftime('%H:%M') for t in timestamps]

axs[0].plot(hour_labels, temperature, label='Actual', marker='o')
axs[0].plot(hour_labels, pred_temperature, label='Predicted', marker='x')
axs[0].set_ylabel("Temperature (°C)")
axs[0].set_title("Hourly Temperature Prediction")
axs[0].legend()

axs[1].plot(hour_labels, visibility, label='Actual', marker='o')
axs[1].plot(hour_labels, pred_visibility, label='Predicted', marker='x')
axs[1].set_ylabel("Visibility (km)")
axs[1].set_title("Hourly Visibility Prediction")
axs[1].legend()

axs[2].plot(hour_labels, uv_index, label='Actual', marker='o')
axs[2].plot(hour_labels, pred_uv_index, label='Predicted', marker='x')
axs[2].set_ylabel("UV Index")
axs[2].set_title("Hourly UV Index Prediction")
axs[2].legend()
axs[2].set_xlabel("Hour")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()