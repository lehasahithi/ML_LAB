import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify the file paths for each class 
class_0_directory = pd.read_csv("C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\DATA\\code_only.csv")
class_1_directory = pd.read_csv("C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\DATA\\code_comm.csv")
class_2_directory = pd.read_csv("C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\DATA\\code_ques.csv")
class_3_directory = pd.read_csv("C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\DATA\\code_sol.csv")

# Extract numpy arrays from DataFrame
feature_name = '0'  
class_0_data = class_0_directory[feature_name]
class_1_data = class_1_directory[feature_name]
class_2_data = class_2_directory[feature_name]
class_3_data = class_3_directory[feature_name]

# Calculate mean and variance for each class
mean_class_0 = np.mean(class_0_data)
variance_class_0 = np.var(class_0_data)

mean_class_1 = np.mean(class_1_data)
variance_class_1 = np.var(class_1_data)

mean_class_2 = np.mean(class_2_data)
variance_class_2 = np.var(class_2_data)

mean_class_3 = np.mean(class_3_data)
variance_class_3 = np.var(class_3_data)

all_data = np.concatenate([class_0_data, class_1_data, class_2_data, class_3_data])

# Calculate overall mean and variance
overall_mean = np.mean(all_data)
overall_variance = np.var(all_data)

# Print results
print("Class 0 - Mean:", mean_class_0, "Variance:", variance_class_0)
print("Class 1 - Mean:", mean_class_1, "Variance:", variance_class_1)
print("Class 2 - Mean:", mean_class_2, "Variance:", variance_class_2)
print("Class 3 - Mean:", mean_class_3, "Variance:", variance_class_3)
print("Overall Mean:", overall_mean)
print("Overall Variance:", overall_variance)
# Plot histograms
plt.hist(class_0_data, bins=20, alpha=0.5, label='Class 0')
plt.hist(class_1_data, bins=20, alpha=0.5, label='Class 1')
plt.hist(class_2_data, bins=20, alpha=0.5, label='Class 2')
plt.hist(class_3_data, bins=20, alpha=0.5, label='Class 3')
plt.title(f'Histogram for Feature {feature_name}')
plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()
