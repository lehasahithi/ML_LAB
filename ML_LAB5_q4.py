import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate synthetic training data
np.random.seed(42)
class1_samples = 100
class2_samples = 100
class1_data = pd.DataFrame({
    'X': np.random.uniform(1, 5, class1_samples),
    'Y': np.random.uniform(1, 5, class1_samples),
    'Class': 'Class1'
})
class2_data = pd.DataFrame({
    'X': np.random.uniform(5, 10, class2_samples),
    'Y': np.random.uniform(5, 10, class2_samples),
    'Class': 'Class2'
})
training_data = pd.concat([class1_data, class2_data], ignore_index=True)

# Generate synthetic test data
test_data = pd.DataFrame({
    'X': np.arange(0, 10.1, 0.1),
    'Y': np.arange(0, 10.1, 0.1),
})

# Create a kNN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the synthetic training data
neigh.fit(training_data[['X', 'Y']], training_data['Class'])

# Classify test data using kNN classifier
predicted_classes_test = neigh.predict(test_data[['X', 'Y']])

# Create a scatter plot of the test data output
plt.scatter(test_data['X'], test_data['Y'], c=np.where(predicted_classes_test == 'Class1', 'blue', 'red'), edgecolors='k', alpha=0.5)

# Add labels and title
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.title('Scatter Plot of Test Data Output')

# Show the plot
plt.show()
