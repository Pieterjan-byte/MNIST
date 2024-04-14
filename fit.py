import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Sample data (replace this with your list of increasing values)
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([0, 44.5829, 78.4911, 144.6647, 248.8494, 401.0412, 625.6398, 906.5131, 1283.7103, 1865.9296])
y_data = np.array([0, 28.6888, 48.7352, 78.9612, 121.7147, 161.0464, 225.8019, 312.2290, 414.1815, 539.4101])  # Sample increasing values
y_data = np.array([0, 24.5941, 40.6174, 69.6882, 112.9669, 162.3049, 217.3319, 299.0830, 380.0930, 480.8399]) 
# Define the functions for quadratic and exponential curves
def quadratic_function(x, a, b, c):
    return a * x ** 2 + b * x + c

def exponential_function(x, a, b):
    return a * np.exp(b * x)

# Fit the quadratic curve
popt_quadratic, pcov_quadratic = curve_fit(quadratic_function, x_data, y_data)

# Fit the exponential curve
popt_exponential, pcov_exponential = curve_fit(exponential_function, x_data, y_data)

# Generate points for the fitted curves
x_fit = np.linspace(min(x_data), max(x_data), 100)

y_fit_quadratic = quadratic_function(x_fit, *popt_quadratic)
y_fit_exponential = exponential_function(x_fit, *popt_exponential)

# Plot the original data and the fitted curves
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_fit, y_fit_quadratic, label='Quadratic Fit', color='red')
plt.plot(x_fit, y_fit_exponential, label='Exponential Fit', color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# Calculate R-squared for each fit
def calculate_r_squared(y_data, y_fit):
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

r_squared_quadratic = calculate_r_squared(y_data, quadratic_function(x_data, *popt_quadratic))
r_squared_exponential = calculate_r_squared(y_data, exponential_function(x_data, *popt_exponential))

print("R-squared for Quadratic Fit:", r_squared_quadratic)
print("R-squared for Exponential Fit:", r_squared_exponential)
