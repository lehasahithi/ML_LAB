import numpy as np
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# Generate random data for features X and Y
X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)

# Generate random class labels (0 or 1) for each data point
classes = np.random.choice([0, 1], size=20)

# Scatter plot for Class 0 points in blue
plt.scatter(X[classes == 0], Y[classes == 0], color='blue', label='Class 0')

# Scatter plot for Class 1 points in red
plt.scatter(X[classes == 1], Y[classes == 1], color='red', label='Class 1')

# Set labels for the axes
plt.xlabel('Feature X')
plt.ylabel('Feature Y')

# Set the title for the plot
plt.title('Scatter Plot of Training Data')

# Display legend to differentiate between classes
plt.legend()

# Show the plot
plt.show()

