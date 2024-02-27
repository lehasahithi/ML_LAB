import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)

classes = np.random.choice([0, 1], size=20)

plt.scatter(X[classes == 0], Y[classes == 0], color='blue', label='Class 0')
plt.scatter(X[classes == 1], Y[classes == 1], color='red', label='Class 1')

plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Scatter Plot of Training Data')

plt.legend()

plt.show()
