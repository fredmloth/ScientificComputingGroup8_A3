import numpy as np
import scipy.linalg

import matplotlib.pyplot as plt

def diagonal_matrix(length):
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

    return M  # Scale by step size squared


# If plots left on a line, prints empty plots, how to remove?
def visualize_multiple_modes(eigenmodes, N, num_modes=6):
    """Plots eigenmodes"""

    horizontal = ((num_modes // 3) +(num_modes % 3))
    fig, axes = plt.subplots(horizontal, 3, figsize=(12, 8))

    # Flatten the axes for correct image rendering
    axes = axes.flatten()
    
    for i in range(num_modes):
        axes[i].imshow(eigenmodes[:, :, i], cmap='bwr', extent=[0, 1, 0, 1])
        axes[i].set_title(f"Mode = {i+1}")
        #axes[i].axis('off')

    #plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()


N = 20
modes = 7

diag_M = diagonal_matrix(N)

# # Each eigenvector column is a mode
eigenvalues, eigenvectors = scipy.linalg.eigh(diag_M)

# # Only take smallest number and each eigenvector column is a mode so 
# # need similar nr of columns 
eigenvalues = eigenvalues[:modes] 
eigenvectors = eigenvectors[:, :modes]
eigenmodes = eigenvectors.reshape(N, N, -1)

visualize_multiple_modes(eigenmodes, N, modes)