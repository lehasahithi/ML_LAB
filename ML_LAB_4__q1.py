import numpy as np
import pandas as pd

# Specify the file paths for each class 
class_0_directory =  pd.read_csv("C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\DATA\\code_only.csv")
class_1_directory = pd.read_csv("C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\DATA\\code_comm.csv")

# Extract numpy arrays from DataFrame
class_0_data = class_0_directory['0']
class_1_data = class_1_directory['1']

# Calculate mean (centroid) for each class
centroid_class_0 = np.mean(class_0_data, axis=0)
centroid_class_1 = np.mean(class_1_data, axis=0)

# Calculate spread (standard deviation) for each class
spread_class_0 = np.std(class_0_data, axis=0)
spread_class_1 = np.std(class_1_data, axis=0)

# Calculate distance between mean vectors (interclass distance)
interclass_distance = np.linalg.norm(centroid_class_0 - centroid_class_1)

# Print results
print("Class 0 centroid:", centroid_class_0)
print("Class 1 centroid:", centroid_class_1)
print("Class 0 spread:", spread_class_0)
print("Class 1 spread:", spread_class_1)
print("Interclass distance between centroids:", interclass_distance)
