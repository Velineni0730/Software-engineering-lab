import numpy as np
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore

# Iteration 4: Optimized Weather Prediction System

# Input time data (0–23 hours)
time = np.arange(0, 24)

# Simulated real-time data
temperature = np.array([
    16, 15, 15, 14, 14, 15, 17, 20, 23, 26, 28, 29,
    30, 29, 27, 25, 24, 22, 21, 20, 19, 18, 17, 16
])
visibility = np.array([
    7, 8, 9, 10, 10, 10, 9, 8, 7, 6, 5, 4,
    3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 7, 7
])
uv_index = np.array([
    0, 0, 0, 1, 2, 3, 5, 7, 9, 10, 11, 11,
    10, 9, 7, 5, 3, 2, 1, 1, 0, 0, 0, 0
])

# Validation
assert len(time) == len(temperature) == len(visibility) == len(uv_index), "Array size mismatch"

# Build quadratic design matrix for efficiency
X = np.vstack((time**2, time, np.ones_like(time))).T

# Solve using least squares
temp_coeffs = np.linalg.lstsq(X, temperature, rcond=None)[0]
vis_coeffs = np.linalg.lstsq(X, visibility, rcond=None)[0]
uv_coeffs = np.linalg.lstsq(X, uv_index, rcond=None)[0]

# Prediction using matrix dot product
t_values = np.arange(0, 24)
X_pred = np.vstack((t_values**2, t_values, np.ones_like(t_values))).T

pred_temp = X_pred @ temp_coeffs
pred_vis = X_pred @ vis_coeffs
pred_uv = X_pred @ uv_coeffs

# Error metrics
mse_temp = mean_squared_error(temperature, X @ temp_coeffs)
mse_vis = mean_squared_error(visibility, X @ vis_coeffs)
mse_uv = mean_squared_error(uv_index, X @ uv_coeffs)

# Print results
print(f"""\nOptimized Quadratic Models:
Temperature:  T(t) = {temp_coeffs[0]:.4f}t² + {temp_coeffs[1]:.4f}t + {temp_coeffs[2]:.4f}
Visibility:   V(t) = {vis_coeffs[0]:.4f}t² + {vis_coeffs[1]:.4f}t + {vis_coeffs[2]:.4f}
UV Index:     UV(t)= {uv_coeffs[0]:.4f}t² + {uv_coeffs[1]:.4f}t + {uv_coeffs[2]:.4f}

MSE (Temperature): {mse_temp:.2f}
MSE (Visibility):  {mse_vis:.2f}
MSE (UV Index):    {mse_uv:.2f}
""")

print("Hour\tTemp (°C)\tVisibility (km)\tUV Index")
for t, temp, vis, uv in zip(t_values, pred_temp, pred_vis, pred_uv):
    print(f"{t:02d}:00\t{temp:.2f}\t\t{vis:.2f}\t\t\t{uv:.2f}")

# Plot all predictions in one figure
plt.figure(figsize=(12, 6))
plt.plot(t_values, pred_temp, 'r--', label='Temp Prediction')
plt.scatter(time, temperature, c='red', s=20, label='Actual Temp')

plt.plot(t_values, pred_vis, 'g--', label='Visibility Prediction')
plt.scatter(time, visibility, c='green', s=20, label='Actual Visibility')

plt.plot(t_values, pred_uv, 'm--', label='UV Index Prediction')
plt.scatter(time, uv_index, c='purple', s=20, label='Actual UV Index')

plt.xlabel("Hour of Day")
plt.ylabel("Value")
plt.title("Optimized Weather Predictions (Temp, Visibility, UV Index)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()