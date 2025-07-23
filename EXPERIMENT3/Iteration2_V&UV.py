

import numpy as np
import matplotlib.pyplot as plt

# Iteration 2: Temperature, Visibility, and UV Index Prediction

# Time values (0 to 23 hours)
time = np.arange(0, 24)

# Sample hourly temperature (°C)
temperature = np.array([
    16, 15, 15, 14, 14, 15, 17, 20, 23, 26, 28, 29,
    30, 29, 27, 25, 24, 22, 21, 20, 19, 18, 17, 16
])

# Sample hourly visibility (in km)
visibility = np.array([
    7, 8, 9, 10, 10, 10, 9, 8, 7, 6, 5, 4,
    3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 7, 7
])

# Sample hourly UV index (0–11 scale)
uv_index = np.array([
    0, 0, 0, 1, 2, 3, 5, 7, 9, 10, 11, 11,
    10, 9, 7, 5, 3, 2, 1, 1, 0, 0, 0, 0
])

# Fit quadratic models
temp_coeffs = np.polyfit(time, temperature, 2)
vis_coeffs = np.polyfit(time, visibility, 2)
uv_coeffs = np.polyfit(time, uv_index, 2)

# Predict values for every hour
t_values = np.arange(0, 24)
pred_temp = temp_coeffs[0]*t_values**2 + temp_coeffs[1]*t_values + temp_coeffs[2]
pred_vis = vis_coeffs[0]*t_values**2 + vis_coeffs[1]*t_values + vis_coeffs[2]
pred_uv = uv_coeffs[0]*t_values**2 + uv_coeffs[1]*t_values + uv_coeffs[2]

# Display equations
print(f"\nTemperature Model: T(t) = {temp_coeffs[0]:.4f}t² + {temp_coeffs[1]:.4f}t + {temp_coeffs[2]:.4f}")
print(f"Visibility Model: V(t) = {vis_coeffs[0]:.4f}t² + {vis_coeffs[1]:.4f}t + {vis_coeffs[2]:.4f}")
print(f"UV Index Model: UV(t) = {uv_coeffs[0]:.4f}t² + {uv_coeffs[1]:.4f}t + {uv_coeffs[2]:.4f}\n")

# Print predictions
print("Hourly Predictions:\n")
for t, temp, vis, uv in zip(t_values, pred_temp, pred_vis, pred_uv):
    print(f"At {t:02d}:00 hrs -> Temp: {temp:.2f} °C | Visibility: {vis:.2f} km | UV Index: {uv:.2f}")

# Plotting
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.plot(t_values, pred_temp, 'r--', label='Predicted Temperature')
plt.scatter(time, temperature, color='blue', label='Original Temperature')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Prediction')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_values, pred_vis, 'g--', label='Predicted Visibility')
plt.scatter(time, visibility, color='orange', label='Original Visibility')
plt.ylabel('Visibility (km)')
plt.title('Visibility Prediction')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_values, pred_uv, 'm--', label='Predicted UV Index')
plt.scatter(time, uv_index, color='purple', label='Original UV Index')
plt.xlabel('Time (Hours)')
plt.ylabel('UV Index')
plt.title('UV Index Prediction')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()