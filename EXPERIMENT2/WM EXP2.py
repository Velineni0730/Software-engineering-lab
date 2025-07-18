import numpy as np
import matplotlib.pyplot as plt

# Step 1: Input Data (Hourly from 0 to 23)

time = np.arange(0, 24, 1)  # Time in hours

temperature = np.array([16, 15, 15, 14, 14, 15, 17, 20, 23, 26, 28, 29, 30, 29, 27, 25, 24, 22, 21, 20, 19, 18, 17, 16])  
# Temperature in °C

humidity = np.array([90, 92, 93, 94, 95, 93, 90, 85, 80, 75, 70, 65, 60, 62, 64, 67, 70, 73, 76, 78, 80, 82, 85, 88]) 
# Humidity in %

wind_speed = np.array([5, 5, 4, 4, 3, 3, 4, 6, 8, 10, 12, 14, 13, 12, 11, 9, 8, 7, 6, 5, 4, 4, 5, 5])  
# Wind speed in km/h

# Step 2: Fit the quadratic model T(t) = a*t^2 + b*t + c

coefficients = np.polyfit(time, temperature, 2)
a, b, c = coefficients
print(f"\nDeveloped Quadratic Model:\nT(t) = {a:.4f}t² + {b:.4f}t + {c:.4f}\n")

# Step 3: Predict temperature for every hour from 0 to 24

t_values = np.arange(0, 25, 1)  # Time from 0 to 24 hours
predicted_temp = a * t_values**2 + b * t_values + c

print("Predicted Temperature (°C) for 24 Hours:\n")
for t, temp in zip(t_values, predicted_temp):
   print(f"At {t:02d}:00 hrs -> {temp:.2f} °C")

# Step 4: Predict humidity for every hour from 0 to 24

h_coeffs = np.polyfit(time, humidity, 2)
predicted_humidity = h_coeffs[0] * t_values**2 + h_coeffs[1] * t_values + h_coeffs[2]

print("\nPredicted Humidity (%) for 24 Hours:\n")
for t, hum in zip(t_values, predicted_humidity):
    print(f"At {t:02d}:00 hrs -> {hum:.2f} %")

# Step 5: Predict wind speed for every hour from 0 to 24

w_coeffs = np.polyfit(time, wind_speed, 2)
predicted_wind = w_coeffs[0] * t_values**2 + w_coeffs[1] * t_values + w_coeffs[2]

print("\nPredicted Wind Speed (km/h) for 24 Hours:\n")
for t, wind in zip(t_values, predicted_wind):
    print(f"At {t:02d}:00 hrs -> {wind:.2f} km/h")

# Step 6: Plotting the results

plt.figure(figsize=(12, 8))

# Temperature Plot
plt.subplot(3, 1, 1)
plt.scatter(time, temperature, color='blue', label='Original Temperature', zorder=5)
plt.plot(t_values, predicted_temp, color='red', linestyle='--', label='Predicted Temperature')
plt.title('Temperature Prediction using Quadratic Model')
plt.xlabel('Time (Hours)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()

# Humidity Plot
plt.subplot(3, 1, 2)
plt.scatter(time, humidity, color='green', label='Original Humidity', zorder=5)
plt.plot(t_values, predicted_humidity, color='orange', linestyle='--', label='Predicted Humidity')
plt.title('Humidity Prediction using Quadratic Model')
plt.xlabel('Time (Hours)')
plt.ylabel('Humidity (%)')
plt.grid(True)
plt.legend()

# Wind Speed Plot
plt.subplot(3, 1, 3)
plt.scatter(time, wind_speed, color='purple', label='Original Wind Speed', zorder=5)
plt.plot(t_values, predicted_wind, color='black', linestyle='--', label='Predicted Wind Speed')
plt.title('Wind Speed Prediction using Quadratic Model')
plt.xlabel('Time (Hours)')
plt.ylabel('Wind Speed (km/h)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()