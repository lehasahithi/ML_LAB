import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

class1_data = pd.read_csv('C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\sem_04\\ML\\DATA\\code_only.csv')
class2_data = pd.read_csv('C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\sem_04\\ML\\DATA\\code_comm.csv')
class1_vectors = class1_data['0'].values
class2_vectors = class2_data['0'].values

class1_vectors = np.array(class1_vectors)
class2_vectors = np.array(class2_vectors)

# Taking two feature vectors from the dataset
vector1 = class1_vectors[0] 
vector2 = class2_vectors[0]

# Ensure the vectors are 1-D
vector1 = np.ravel(vector1)
vector2 = np.ravel(vector2)

# Initialize lists to store distances and r values
distances = []
r_values = list(range(1, 11))

# Calculate Minkowski distance for each value of r
for r in r_values:
    distance = minkowski(vector1, vector2, p=r)
    distances.append(distance)

# Plotting the distances
plt.plot(r_values, distances, marker='o', linestyle='-')
plt.xlabel('r')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance between Two Feature Vectors')
plt.grid(True)
plt.show()

# Creating class labels
class1_labels = ['Class1'] * len(class1_vectors)
class2_labels = ['Class2'] * len(class2_vectors)

# Combine feature vectors and class labels
class1_df = pd.DataFrame({'Feature': class1_vectors, 'Class': class1_labels})
class2_df = pd.DataFrame({'Feature': class2_vectors, 'Class': class2_labels})

# Combine both classes into one DataFrame
data = pd.concat([class1_df, class2_df], ignore_index=True)

# Separate features (X) and labels (y)
X = data['Feature']
y = data['Class']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Ensure the shape of the train and test sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Create a kNN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
neigh.fit(X_train.values.reshape(-1, 1), y_train) 

# Test the accuracy of the kNN classifier on the test set
accuracy = neigh.score(X_test.values.reshape(-1, 1), y_test)

print("Accuracy of the kNN classifier on the test set:", accuracy)

# Use the predict() function to predict classes for the test set
predicted_classes = neigh.predict(X_test.values.reshape(-1, 1))

# Print the predicted classes
print("Predicted classes for the test set:")
print(predicted_classes)

# Initialize lists to store accuracy values for different values of k
accuracies = []

# Vary k from 1 to 11
for k in range(1, 12):
    # Create kNN classifier with current k value
    neigh = KNeighborsClassifier(n_neighbors=k)
    
    # Train the classifier on the training data
    neigh.fit(X_train.values.reshape(-1, 1), y_train)
    
    # Test the accuracy on the test set
    accuracy = neigh.score(X_test.values.reshape(-1, 1), y_test)
    
    # Append accuracy to the list
    accuracies.append(accuracy)

# Plotting the accuracy values
plt.plot(range(1, 12), accuracies, marker='o', linestyle='-')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy of kNN for different values of k')
plt.xticks(range(1, 12))
plt.grid(True)
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Precision, recall, and F1-score
precision = precision_score(y_test, predicted_classes, average='weighted')
recall = recall_score(y_test, predicted_classes, average='weighted')
f1 = f1_score(y_test, predicted_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Confusion matrix for training data
train_predicted_classes = neigh.predict(X_train.values.reshape(-1, 1))
conf_matrix_train = confusion_matrix(y_train, train_predicted_classes)
print("\nConfusion Matrix (Training Data):")
print(conf_matrix_train)

# Precision, recall, and F1-score for training data
precision_train = precision_score(y_train, train_predicted_classes, average='weighted')
recall_train = recall_score(y_train, train_predicted_classes, average='weighted')
f1_train = f1_score(y_train, train_predicted_classes, average='weighted')

print("Precision (Training Data):", precision_train)
print("Recall (Training Data):", recall_train)
print("F1-Score (Training Data):", f1_train)

# Precision, recall, and F1-score for test data
precision_test = precision_score(y_test, predicted_classes, average='weighted')
recall_test = recall_score(y_test, predicted_classes, average='weighted')
f1_test = f1_score(y_test, predicted_classes, average='weighted')

print("\nPrecision (Test Data):", precision_test)
print("Recall (Test Data):", recall_test)
print("F1-Score (Test Data):", f1_test)

# Model learning outcome inference
# Check if the model is underfit, regularfit, or overfit
if accuracy == 1.0:
    print("\nThe model may be overfitting the training data.")
elif accuracy == 0.0:
    print("\nThe model may be underfitting the training data.")
else:
    print("\nThe model is fitting well to the data.")
    if accuracy > max(accuracies):
        print("The model may be overfitting.")
    elif accuracy < max(accuracies):
        print("The model may be underfitting.")
    else:
        print("The model is fitting well to the data.")

# Assuming you have a column named 'actual_prices' in your DataFrame for true prices
actual_prices = df['Price'].values  

# Assuming you have a column named 'regression_predictions' in your DataFrame for predicted prices
regression_predictions = df['Price'].values

# Calculate MSE
mse = mean_squared_error(actual_prices, regression_predictions)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAPE
mape = np
