import numpy as np
import scipy.linalg

import matplotlib.pyplot as plt

def diagonal_matrix(length):
    """Creates the Matrix M of the eigenvalue problem. Length is the 
    (length x length) size of a grid."""

    N = length*length

    # Create zero matrix, size of (N*N) x (N*N) 
    matrix_M = np.zeros((N, N))

    # i == j
    np.fill_diagonal(matrix_M, -4)

    # i == j-1
    np.fill_diagonal(matrix_M[1:], 1)

    # i == j+1
    np.fill_diagonal(matrix_M[:, 1:], 1)

    # i == j-4
    np.fill_diagonal(matrix_M[4:], 1)

    # i == j+4
    np.fill_diagonal(matrix_M[:, 4:], 1)

    return matrix_M

N = 4
modes = 6

diag_M = diagonal_matrix(N)
print(diag_M)

# Each eigenvector column is a mode

eigenvalues, eigenvectors = scipy.linalg.eigh(diag_M)

# Only take smallest number and each eigenvector column is a mode so 
# need similar nr of columns 
smallest_nr = 6
eigenvalues = eigenvalues[:modes] 
eigenvectors = eigenvectors[:, :modes]

print(eigenvalues)

