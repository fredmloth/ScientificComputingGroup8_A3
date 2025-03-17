import numpy as np
import scipy.linalg

import matplotlib.pyplot as plt

def diagonal_matrix(N):
    """Creates the Matrix M of the eigenvalue problem. Length is the 
    (length x length) size of a grid."""

    size = N * N

    # Initialize matrix with zeros
    M = np.zeros((size, size))

    for i in range(size):
        M[i, i] = -4  # Center point

        if (i + 1) % N != 0:  # Right neighbor
            M[i, i + 1] = 1
        
        if i % N != 0:  # Left neighbor
            M[i, i - 1] = 1

        if i + N < size:  # Bottom neighbor
            M[i, i + N] = 1

        if i - N >= 0:  # Top neighbor
            M[i, i - N] = 1

    for n in range(N, size, N):
         M[n, n-1] = 0
         M[n-1, n] = 0 


    return M  # Scale by step size squared


def visualize_diag_matrix(M, N, text='ON'):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.matshow(M, cmap='viridis')

    # Should turn off for higher values
    if text == 'ON':
        for i in range(N*N):
            for j in range(N*N):
                c = M[j, i]
                if c == -4:
                    ax.text(i, j, int(c), va='center', ha='center', c='white')
                elif c == 1:
                    ax.text(i, j, int(c), va='center', ha='center')

    return fig


def visualize_multiple_modes(eigenmodes, N, num_modes=6):
    """Plots eigenmodes"""

    horizontal = ((num_modes // 3) +(num_modes % 3))
    fig, axes = plt.subplots(horizontal, 3, figsize=(12, 8))

    # Flatten the axes for correct image rendering
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()
    
    for i in range(num_modes):
        axes[i].set_axis_on()
        axes[i].imshow(eigenmodes[:, :, i], cmap='bwr', extent=[0, 1, 0, 1])
        axes[i].set_title(f"Mode = {i+1}")
        #axes[i].axis('off')

    #plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()


def matrix_grid(grid):
    """If we need to create a grid for the problem"""
    return np.pad(grid, 1)

def matrix_vector(matrix, method='row'):
    """If wee need to make a vector from a grid with boundary conditions"""
    if method == 'row':
        vector = matrix.reshape(1, -1)

    elif method == 'column':
        vector = matrix.reshape(1, -1)

    elif method == '1D array':
        vector = matrix.ravel()

    else:
        raise ValueError("Incorrect method parameter, choose: 'row', 'column'"
            " or '1S array'.")

    return vector


def get_eigenmodes(M, N, modes):
    # # Each eigenvector column is a mode
    eigenvalues, eigenvectors = scipy.linalg.eigh(M)

    # # Only take smallest number and each eigenvector column is a mode so 
    # # need similar nr of columns 
    eigenvalues = eigenvalues[:modes] 
    eigenvectors = abs(eigenvectors[:, :modes])
    eigenmodes = eigenvectors.reshape(N, N, -1)

    return eigenmodes

# N = 4
# modes = 6

# diag_M = diagonal_matrix(N)
# fig = visualize_diag_matrix(diag_M, N)

# eigenmodes = get_eigenmodes(diag_M, modes)
# visualize_multiple_modes(eigenmodes, N, modes)


