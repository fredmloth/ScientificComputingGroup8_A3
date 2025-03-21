import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def diagonal_matrix(N):
    """Creates the Matrix M of the eigenvalue problem. Length is the 
    (length x length) size of a grid."""

    size = N * N
    dx = (1 / N)

    # Initialize matrix with zeros
    M = np.zeros((size, size))

    for i in range(size):
        M[i, i] = (-4)  # Center point

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

    return M / (dx**2)


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


def visualize_multiple_modes(eigenmodes, eigenvalues, N, num_modes=6):
    """Plots eigenmodes"""

    horizontal = ((num_modes // 3) +(num_modes % 3))
    fig, axes = plt.subplots(horizontal, 3, figsize=(12, 8))

    # Flatten the axes for correct image rendering
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()
    
    for i in range(num_modes):
        max = np.max(np.abs(eigenmodes[:,:,i]))
        axes[i].set_axis_on()
        axes[i].imshow(eigenmodes[:, :, i], cmap='bwr', extent=[0, 1, 0, 1], vmin = -max, vmax = max)
        axes[i].set_title(f"Mode = {i+1}, Eigenvalue = {eigenvalues[i]}")
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


def get_eigenmodes(M, N, modes=6):

    dx = (1 / N)
    M /= dx**2

    # Each eigenvector column is a mode
    eigenvalues, eigenvectors = scipy.linalg.eigh(M)
    sorted_eig = np.argsort(np.abs(eigenvalues))[:modes]

    # Only take smallest number and each eigenvector column is a mode so 
    # need similar nr of columns
    eigenvectors = eigenvectors[:, sorted_eig]
    eigenvalues = eigenvalues[sorted_eig]
    eigenmodes = eigenvectors.reshape(N, N, -1)
    
    return eigenvalues, eigenvectors, sorted_eig, eigenmodes


def rectangular_domain(L):
    """The rectangle's 2L side is assumed to be horizontal"""
    if L % 2 != 0:
        raise ValueError("L must be an integer divided by two, improper "
            "borders.")
    
    grid = np.zeros((L, 2*L))
    grid[:, :] = False
    L_half = int(L / 2)
    grid[L_half:-L_half, :] = True

    return grid


def circular_domain(N):
    """Circular domain according to the leuclidian domain.
    If L is even, the center (L//2, L//2) falls on the corner of four pixels.
    If L is odd, the center is directly on a single pixel, making the circle more symmetric.
    """

    grid = np.zeros((N, N))
    grid[:, :] = False
    
    radius = (N / 2)
    center = (N // 2, N//2)

    y,x = np.ogrid[:N, :N]

    mask = (x-center[0])**2 + (y-center[1])**2 <= radius**2

    grid[mask] = True
    return grid


def test_domain(grid):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.matshow(grid, cmap='viridis')

    plt.show()
    return
    

def diagonal_circle(grid, diag_M, N):
    """Filters the diagonal for a matrix that is not a square"""
    for i in range(N):
        for j in range(N):
            if grid[i][j] == False:
                index = i * N + j

                diag_M[index, :] = 3
                diag_M[:, index] = 3

    # Flip True/False to remove correct cols and rows
    rows_to_keep = ~np.all(diag_M == 3, axis=1)
    cols_to_keep = ~np.all(diag_M == 3, axis=0)

    filtered_matrix = diag_M[rows_to_keep, :][:, cols_to_keep]

    return filtered_matrix


def diagonal_rectangular(L):
    N = (2*L)
    size = L * N

    M = np.zeros((size, size))

    for i in range(size):
        M[i, i] = -4  # Center point

        if (i + 1) % N != 0:  # Right 
            M[i, i + 1] = 1
        
        if i % N != 0:  # Left 
            M[i, i - 1] = 1

        if i + N < size:  # Bottom 
            M[i, i + N] = 1

        if i - N >= 0:  # Top 
            M[i, i - N] = 1

    for n in range(N, size, N):
        M[n, n-1] = 0
        M[n-1, n] = 0 

    return M


def get_eigenmodes_rectangular(M, N, modes=6):
    """Eigenmodes for rectangle"""
    dx = (1 / 2*N)
    M /= dx**2

    eigenvalues, eigenvectors = scipy.linalg.eigh(M)
    sorted_eig = np.argsort(np.abs(eigenvalues))[:modes]
    eigenvectors = eigenvectors[:, sorted_eig]
    eigenvalues = eigenvalues[sorted_eig]

    # Reshape eigenmodes for rectangule (L x 2L)
    eigenmodes = eigenvectors.reshape(N, 2*N, -1)
    
    return eigenvalues, eigenvectors, sorted_eig, eigenmodes


def get_eigenmodes_circular(M, grid, N, modes=6):
    """Eigenmodes for circular grid"""
    dx = (1 / N)
    M /= dx**2

    eigenvalues, eigenvectors = scipy.linalg.eigh(M)
    sorted_eig = np.argsort(np.abs(eigenvalues))[:modes]
    eigenvectors = eigenvectors[:, sorted_eig]
    eigenvalues = eigenvalues[sorted_eig]

    # Create an empty 3D array to store eigenmodes in the circular shape
    eigenmodes = np.zeros((N, N, modes))

    # Get indexes of True grid
    indexes = np.where(grid)

    for i in range(modes):
        mode_vector = eigenvectors[:, i]
        mode_grid = np.zeros((N, N))
        mode_grid[indexes] = mode_vector

        eigenmodes[:, :, i] = mode_grid
    
    return eigenvalues, eigenvectors, sorted_eig, eigenmodes

def get_eigenmodes_sparse_square(M, N, modes=6):
    """Eigenmodes for sparse matrix"""
    dx = (1 / N)
    M /= dx**2

    M_sparse = scipy.sparse.csr_matrix(M)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(M_sparse, 
        k=modes, which='SM')

    eigenmodes = eigenvectors.reshape(N, N, -1)

    return eigenvalues, eigenvectors, eigenmodes

def get_eigenmodes_sparse_rectangular(M, N, modes=6):
    """Eigenmodes for sparse matrix"""
    dx = (1 /2*N)
    M /= dx**2

    M_sparse = scipy.sparse.csr_matrix(M)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(M_sparse, 
        k=modes, which='SM')

    # Reshape eigenmodes for rectangle (L x 2L)
    eigenmodes = eigenvectors.reshape(N, 2 * N, -1)

    return eigenvalues, eigenvectors, eigenmodes

def get_eigenmodes_sparse_circular(M, grid, N, modes=6):
    dx = (1 / N)
    M /= dx**2

    M_sparse = scipy.sparse.csr_matrix(M)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(M_sparse, 
        k=modes, which='SM')

    eigenmodes = np.zeros((N, N, modes))
    indexes = np.where(grid)

    for i in range(modes):
        mode_vector = eigenvectors[:, i]
        mode_grid = np.zeros((N, N))
        mode_grid[indexes] = np.real(mode_vector)
        eigenmodes[:, :, i] = mode_grid

    return eigenvalues, eigenvectors, eigenmodes

def eigenfreqs_lengths_square(L_lengths, modes=6):
    L_eigenfreqs = []
    for length in L_lengths:
        diag_M = diagonal_matrix(length)

        eigenfreqs, _, _ = get_eigenmodes_sparse_square(diag_M, length, modes)
        L_eigenfreqs.append(eigenfreqs)
    return L_eigenfreqs
        

def eigenfreqs_lengths_rectangle(L_lengths,modes=6):
    L_eigenfreqs = []
    for length in L_lengths:
        diag_M = diagonal_matrix(length)

        eigenfreqs, _, _ = get_eigenmodes_sparse_rectangular(diag_M, length, modes)
        L_eigenfreqs.append(eigenfreqs)
    return L_eigenfreqs

def eigenfreqs_lengths_circular(L_lengths,modes=6):
    L_eigenfreqs = []
    for length in L_lengths:
        diag_M = diagonal_matrix(length)
        grid = circular_domain(length)
        diag_M = diagonal_circle(grid, diag_M, length) 

        eigenfreqs, _, _ = get_eigenmodes_sparse_circular(diag_M, grid, length, modes)
        L_eigenfreqs.append(eigenfreqs)
    return L_eigenfreqs

def visualise_eigenfreqs_lengths(lengths, LL_eigenfreqs):
    """Plots the eigenfrequencies for different values of N for all three
    domain shapes.
    
    Parameters:
    - lengths (list[int]): The list of lengths for which the eigenfrequencies
      have been calculated.
    - LL_eigenfreqs (list[list[float]]): A list with the calculated
      eigenfrequencies for each of the three shapes"""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Flatten the axes for correct image rendering
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()
    
    L_lengths = [[x]*6 for x in lengths]
    domain_shapes = ["Square", "Rectangular", "Circular"]
    for i in range(len(domain_shapes)):
        axes[i].set_axis_on()
        for j in range(len(L_lengths)):
            axes[i].scatter(L_lengths[j], LL_eigenfreqs[i][j], alpha=0.5)
        axes[i].set_xlim(0,max(lengths)+10)
        axes[i].set_ylim(-150,0)
        axes[i].set_xlabel("length")
        axes[i].set_ylabel("eigenfrequencies")
        axes[i].set_title(f"Shape = {domain_shapes[i]}")
        #axes[i].axis('off')

    #plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()

def time_dependent_visualise_square(eigenmode, eigenval, time=1, num_times=4, A=1, B=1, c=1):
    max = np.max(np.abs(eigenmode))

    horizontal = ((num_times // 3) +(num_times % 3))
    fig, axes = plt.subplots(horizontal, 3, figsize=(12, 4*horizontal))
    
    # Flatten the axes for correct image rendering
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()

    for i in range(num_times):
        t = i*(time/num_times)
        u = eigenmode * (A*np.cos(c*eigenval*t)+B*np.sin(c*eigenval*t))
        axes[i].imshow(u, cmap="bwr", vmin=-max, vmax=max)
        axes[i].set_axis_on()
        axes[i].set_title(f"t={t}")

    plt.tight_layout()
    plt.show()

def time_dependent_animation_square(eigenmode, eigenval, time=1, step=0.01, A=1, B=1, c=1):
    """"""
    max = np.max(np.abs(eigenmode))

    fig, ax = plt.subplots()
    im = ax.imshow(eigenmode, cmap="bwr", vmin=-max, vmax=max)

    # Store precomputed frames
    ims = []
    for t in np.arange(0+step, time, step):
        u = eigenmode * (A*np.cos(c*eigenval*t)+B*np.sin(c*eigenval*t))
        im_ = ax.imshow(u, cmap="bwr", animated=True, vmin=-max, vmax=max)
        ims.append([im_])

    # Use ArtistAnimation
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

    # Close the figure to not display in the notebook
    plt.close(fig)

    return ani

def visualise():
    return
