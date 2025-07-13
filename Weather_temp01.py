import numpy as np

# Step 1: Input Data

time = np.array([0, 4, 8, 12, 16, 20])  # Time in hours

temperature = np.array([15, 18, 24, 29, 25, 19])  # Temperature in °C

 

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