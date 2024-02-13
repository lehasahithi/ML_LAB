import pandas as pd
import numpy as np
from numpy.linalg import pinv

data = pd.read_excel("C:\\Users\\lehas\\OneDrive\\Documents\\B-TECH_AIE_D\\sem_04\\ML\\Lab Session1 Data.xlsx", sheet_name="Purchase data")
print(data)

A = data[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
C = data[['Payment (Rs)']].values 

print("Matrix A:\n", A)
print("Matrix C:\n", C)

dimensionality = A.shape[1]

num_vectors = A.shape[0]

rank_A = np.linalg.matrix_rank(A)
pseudo_inverse_A = pinv(A)
cost_per_product = np.dot(pseudo_inverse_A, C)

print("Dimensionality of the vector space:", dimensionality)
print("Number of vectors:", num_vectors)
print("Rank of Matrix A:", rank_A)
print("Cost of each product available for sale:", cost_per_product)
