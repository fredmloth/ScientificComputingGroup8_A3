import numpy as np
import scipy.linalg

def laplacian_matrix(N):

    # Create zero matrix, size of (N*N) x (N*N) 
    matrix_M = np.zeros((N*N, N*N))

    # Create diagonals from 5-point-stencil
    for i in range(N):
        for j in range(N):
        
            if i == j:
                matrix_M[i,j] = -4
    return