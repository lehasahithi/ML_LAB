def multiply_matrices(A, B):
    result = []
#for the multipling the matrices
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
          
            value = sum(A[i][k] * B[k][j] for k in range(len(B)))
            row.append(value)
        result.append(row)

    return result

def matrix_power(A,m):
#for the matrix to the power  multiplication 
    result = A
    for _ in range(m - 1):
        result = multiply_matrices(result, A)
    return result

if __name__ == "__main__":
   
    input_matrix = [[1, 2],
                    [2, 1]]

    power = 2
    result_matrix = matrix_power(input_matrix, power)

    print(f"The matrix raised to the power of {power}:\n{result_matrix}")
