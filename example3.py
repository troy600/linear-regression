import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([3, 4, 6, 9, 45, 22])

# Linear regression
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]

# Plotting
plt.plot(x, y, 'bo', label='Data points')
plt.plot(x, m*x + c, 'r', label=f'Linear fit: y = {m:.2f}x + {c:.2f}')
plt.legend()
plt.show()

print(f"Equation: y = {m:.2f}x + {c:.2f}")

