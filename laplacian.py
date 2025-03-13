import numpy as np
import scipy.linalg

def laplacian_matrix(length):

    N = length*length

    # Create zero matrix, size of (N*N) x (N*N) 
    matrix_M = np.zeros((N, N))

    # Create diagonals from 5-point-stencil
    for i in range(N):
        for j in range(N):
        
            if i == j:
                matrix_M[i,j] = -4

            if i == (j-1):
                matrix_M[i,j] = 1
    print(matrix_M)
    return

laplacian_matrix(4)