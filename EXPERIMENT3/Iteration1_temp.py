import numpy as np
import matplotlib.pyplot as plt

# Iteration 1: Basic Model - Temperature Prediction

# Step 1: Input time (hours) and temperature data
time = np.array([0, 4, 8, 12, 16, 20])  # Time in hours
temperature = np.array([16, 18, 24, 29, 25, 20])  # Temperature in °C

# Step 2: Fit quadratic model T(t) = a*t^2 + b*t + c
coeffs = np.polyfit(time, temperature, 2)
a, b, c = coeffs
print(f"\nQuadratic Model: T(t) = {a:.4f}t² + {b:.4f}t + {c:.4f}\n")

# Step 3: Predict temperature for each hour 0–24
t_values = np.arange(0, 25)
predicted_temp = a * t_values**2 + b * t_values + c

# Step 4: Print predicted values
print("Predicted Temperature (°C) for 24 Hours:")
for t, temp in zip(t_values, predicted_temp):
    print(f"At {t:02d}:00 hrs -> {temp:.2f} °C")

# Step 5: Plot original vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(time, temperature, color='blue', label='Original Data')
plt.plot(t_values, predicted_temp, color='red', linestyle='--', label='Predicted')
plt.title('Iteration 1: Temperature Prediction using Quadratic Model')
plt.xlabel('Time (Hours)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()