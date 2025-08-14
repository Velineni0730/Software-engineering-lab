import numpy as np
import os

# ---------------- Stage 1: Hard-coded variables ----------------
print("Stage 1 — Hard-coded variables:")
time = np.array([0, 4, 8, 12, 16, 20])
temperature = np.array([15, 18, 24, 29, 25, 19])

# Fit quadratic model
coefficients = np.polyfit(time, temperature, 2)
a, b, c = coefficients
print(f"Quadratic Model: T(t) = {a:.4f}t² + {b:.4f}t + {c:.4f}\n")

# ---------------- Stage 2: Keyboard input ----------------
print("Stage 2 — Keyboard input:")
time_input = list(map(float, input("Enter time values (space-separated): ").split()))
temp_input = list(map(float, input("Enter temperature values (space-separated): ").split()))

coeffs_input = np.polyfit(time_input, temp_input, 2)
print(f"Quadratic Model: T(t) = {coeffs_input[0]:.4f}t² + {coeffs_input[1]:.4f}t + {coeffs_input[2]:.4f}\n")

# ---------------- Stage 3: Read from file ----------------
# File format (weather_data.txt):
# First line: space-separated time values
# Second line: space-separated temperature values
print("Stage 3 — Reading from file:")
file_path = os.path.abspath("weather_data.txt")
try:
    with open(file_path, "r") as f:
        lines = f.readlines()
        time_file = list(map(float, lines[0].split()))
        temp_file = list(map(float, lines[1].split()))
        coeffs_file = np.polyfit(time_file, temp_file, 2)
        print(f"Quadratic Model: T(t) = {coeffs_file[0]:.4f}t² + {coeffs_file[1]:.4f}t + {coeffs_file[2]:.4f}\n")
except (FileNotFoundError, IndexError, ValueError):
    print(f"{file_path} not found or invalid format. Using default hard-coded values.\n")
    time_file = [0, 4, 8, 12, 16, 20]
    temp_file = [15, 18, 24, 29, 25, 19]
    coeffs_file = np.polyfit(time_file, temp_file, 2)
    print(f"Quadratic Model: T(t) = {coeffs_file[0]:.4f}t² + {coeffs_file[1]:.4f}t + {coeffs_file[2]:.4f}\n")

# ---------------- Stage 4: Single set of input ----------------
print("Stage 4 — Single set of prediction:")
t_single = float(input("Enter the hour (0-24) to predict temperature: "))
predicted_temp_single = a * t_single**2 + b * t_single + c
print(f"Predicted temperature at {t_single} hrs: {predicted_temp_single:.2f} °C\n")

# ---------------- Stage 5: Multiple sets of input ----------------
print("Stage 5 — Multiple predictions for 0 to 24 hrs:")
t_values = np.arange(0, 25, 1)
predicted_temp_multiple = a * t_values**2 + b * t_values + c
for t, temp in zip(t_values, predicted_temp_multiple):
    print(f"At {t:02d}:00 hrs -> {temp:.2f} °C")