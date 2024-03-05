import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer


file_path = 'C:/Users/91630/Downloads/code_only.csv'
your_data = pd.read_csv(file_path, header=None, dtype=str)  

# Extract the target variable (y) and convert it to float
y = your_data.iloc[:, 0].astype(float)  

# Convert the remaining features to numeric
X = your_data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Set a threshold to create binary labels (0 or 1) based on the mean of y
threshold = y.mean()  
y_binary = (y > threshold).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Impute missing values in the features using the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create a KNeighborsClassifier instance
knn = KNeighborsClassifier()

# Define the hyperparameter grid for tuning
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}

# Use GridSearchCV to find the best hyperparameters through cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_imputed, y_train)

# Display the best hyperparameters found by GridSearchCV
print("Best hyperparameters:", grid_search.best_params_)

# Get the best KNeighborsClassifier based on the tuned hyperparameters
best_knn = grid_search.best_estimator_

# Evaluate the accuracy on the test set using the best model
accuracy = best_knn.score(X_test_imputed, y_test)
print("Test set accuracy:", accuracy)
