"""This module contains the eigenvalue problem solutions."""

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def diagonal_matrix(N):
    """Creates and returns the discretized laplacian Matrix M of size 
    (N^2 by N^2) of the eigenvalue problem. N is the (NxN) size of a 
    grid."""

    size = N * N

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

    return M 


def diagonal_rectangular(L):
    """Creates and returns the discretized laplacian Matrix M of the 
    eigenvalue problem. L is the length of a grid (Lx2L). Matrix M is of 
    size (2L^2 by 2L^2)."""
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


def diagonal_rectangle(grid, diag_M, L):
    """
    Removes rows from the diagonal matrix outside of the rectangular 
    grid bounds.

    Args:
        grid (np.ndarray): Boolean array of shape (L, 2*L).
        diag_m (np.ndarray): Matrix to mark and filter.
        L (int): Expected number of rows in grid.

    Returns:
        np.ndarray: Filtered version of diag_m.
    """
    rows, cols = grid.shape  # should be (L, 2L)
    assert rows == L and cols == 2*L, "grid must be shape (L, 2L)"
    
    for i in range(rows):
        for j in range(cols):
            if not grid[i, j]:
                idx = i*cols + j
                # Mark row & column idx
                diag_M[idx, :] = 3
                diag_M[:, idx] = 3

    # Identify rows/columns that are not fully marked
    rows_to_keep = ~np.all(diag_M == 3, axis=1)
    cols_to_keep = ~np.all(diag_M == 3, axis=0)

    # Filter out fully marked rows and columns
    filtered_matrix = diag_M[rows_to_keep, :][:, cols_to_keep]
    
    return filtered_matrix


def diagonal_circle(grid, diag_M, N):
    """
    Removes rows from the diagonal matrix outside of the circular grid 
    bounds.

    Args:
        grid (np.ndarray): Boolean array of shape (N, N).
        diag_M (np.ndarray): Matrix to mark and filter.
        N (int): Width and height of the grid.

    Returns:
        np.ndarray: Filtered version of diag_M.
    """
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


def visualize_diag_matrix(M, N):
    """
    Visualizes a diagonal matrix using matplotlib.

    Args:
        M (np.ndarray): Square matrix of shape (N*N, N*N) to visualize.
        N (int): Grid size used to interpret the shape.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
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


def visualize_multiple_modes(eigenmodes, eigenvalues, L, num_modes=6):
    """
    Plots the first few eigenmodes with corresponding eigenvalues.

    Args:
        eigenmodes (np.ndarray): 3D array of shape (L, L, num_modes).
        eigenvalues (np.ndarray): 1D array of eigenvalues.
        L (int): Size of each eigenmode (assumed square).
        num_modes (int): Number of modes to plot (default is 6).

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    horizontal = ((num_modes // 3) +(num_modes % 3))
    fig, axes = plt.subplots(horizontal, 3, figsize=(12, 8))

    # Flatten the axes for correct image rendering
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()
    
    for i in range(num_modes):
        max = np.max(np.abs(eigenmodes[:,:,i]))
        axes[i].set_axis_on()
        axes[i].imshow(eigenmodes[:, :, i], cmap='bwr', extent=[0, L, 0, L], 
            vmin = -max, vmax = max)
        axes[i].set_title(f"Mode = {i+1}, "
            f"Eigenvalue = {round(eigenvalues[i],2)}")

    plt.tight_layout()
    plt.close(fig)

    return fig


def visualize_rectangular_modes(eigenmodes, eigvals, L, modes=6):
    """
    Plots eigenmodes for a rectangular domain (2L wide, L high) and 
    saves the figure.

    Args:
        eigenmodes (np.ndarray): 3D array of shape (L, 2*L, modes).
        eigvals (np.ndarray): 1D array of eigenvalues.
        L (int): Half the width of the mode shape (total width = 2*L).
        modes (int): Number of modes to plot (default is 6).

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    horizontal = ((modes // 3) +(modes % 3))
    fig, axes = plt.subplots(horizontal, 3, figsize=(10, 6), squeeze=False)
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()

    for i in range(modes):
        vmax = np.max(np.abs(eigenmodes[:,:,i]))

        # Show an image that is 2 wide, 1 high
        im = axes[i].imshow(eigenmodes[:,:,i], cmap='bwr', extent=[0, 2*L, 0, L],
            vmin=-vmax, vmax=vmax, aspect='equal')
        axes[i].set_axis_on()
        axes[i].set_title(f"Mode {i+1} λ = {eigvals[i]:.2f}")

    plt.tight_layout()
    plt.close(fig)
    
    return fig


def circular_domain(N):
    """
    Generates a circular domain on an N x N grid.

    If N is even, the center lies between four pixels.
    If N is odd, the center aligns with a single pixel.

    Args:
        N (int): Size of the square grid.

    Returns:
        np.ndarray: Boolean mask with True inside the circle, 
            False outside.
    """
    grid = np.zeros((N, N))
    grid[:, :] = False
    
    radius = (N / 2)
    center = (N // 2, N//2)

    y,x = np.ogrid[:N, :N]

    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    grid[mask] = True

    return grid
    

def get_eigenmodes(M, L, dx, modes=6):
    """
    Computes the lowest eigenmodes of a matrix.

    Args:
        M (np.ndarray): Square matrix to diagonalize.
        L (float): Physical length of the domain.
        dx (float): Grid spacing.
        modes (int): Number of lowest modes to return (default is 6).

    Returns:
        tuple:
            - eigenvalues (np.ndarray): Selected eigenvalues, scaled by dx².
            - eigenvectors (np.ndarray): Corresponding eigenvectors.
            - sorted_eig (np.ndarray): Indices of selected eigenvalues.
            - eigenmodes (np.ndarray): Reshaped eigenvectors as (steps, steps, modes).
    """
    steps = int(L / dx)

    # Compute all eigenvalues and eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(M)

    # Select indices of the smallest eigenvalues
    sorted_eig = np.argsort(np.abs(eigenvalues))[:modes]

    # Filter the corresponding eigenvectors and eigenvalues
    eigenvectors = eigenvectors[:, sorted_eig]
    eigenvalues /= dx**2
    eigenvalues = eigenvalues[sorted_eig]

    # Reshape eigenvectors to 2D modes
    eigenmodes = eigenvectors.reshape(steps, steps, -1)
    
    return eigenvalues, eigenvectors, sorted_eig, eigenmodes


def get_eigenmodes_rectangular(M, L, dx, modes=6):
    """
    Computes eigenmodes for a rectangular domain (2L wide, L high).

    Args:
        M (np.ndarray): Matrix to diagonalize.
        L (float): Half the width of the domain.
        dx (float): Grid spacing.
        modes (int): Number of modes to return (default is 6).

    Returns:
        tuple:
            - eigvals (np.ndarray): Selected eigenvalues, scaled by dx².
            - eigvecs (np.ndarray): Corresponding eigenvectors.
            - sorted_eig (np.ndarray): Indices of selected eigenvalues.
            - eigenmodes (np.ndarray): Reshaped eigenmodes of shape 
            (steps, 2*steps, modes).
    """
    steps = int(L/dx)

    # Solve (real symmetric) for all eigenvalues
    eigvals, eigvecs = scipy.linalg.eigh(M)

    # Sort by smallest magnitude (or smallest negative, depending on sign)
    sorted_eig = np.argsort(np.abs(eigvals))[:modes]
    eigvals = eigvals[sorted_eig]
    eigvecs = eigvecs[:, sorted_eig]
    eigvals /= dx**2

    # Reshape to rectangular domain (2L wide, L high)
    eigenmodes = eigvecs.reshape(steps, 2*steps, -1)

    return eigvals, eigvecs, sorted_eig, eigenmodes


def get_eigenmodes_circular(M, grid, L, dx, modes=6):
    """
    Computes eigenmodes for a circular domain.

    Args:
        M (np.ndarray): Matrix to diagonalize.
        grid (np.ndarray): Boolean mask for circular domain shape.
        L (float): Physical size of the square grid.
        dx (float): Grid spacing.
        modes (int): Number of modes to return (default is 6).

    Returns:
        tuple:
            - eigenvalues (np.ndarray): Selected eigenvalues, scaled by 
              dx².
            - eigenvectors (np.ndarray): Corresponding eigenvectors.
            - sorted_eig (np.ndarray): Indices of selected eigenvalues.
            - eigenmodes (np.ndarray): Modes reshaped to (steps, steps, 
              modes),
              with values only in the circular region.
    """
    steps = int(L/dx)

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = scipy.linalg.eigh(M)

    # Sort and select lowest-magnitude eigenvalues
    sorted_eig = np.argsort(np.abs(eigenvalues))[:modes]
    eigenvectors = eigenvectors[:, sorted_eig]
    eigenvalues /= dx**2
    eigenvalues = eigenvalues[sorted_eig]

    # Prepare empty for spatial eigenmodes
    eigenmodes = np.zeros((steps, steps, modes))

    # Insert each eigenvector into the circular grid
    indexes = np.where(grid)
    for i in range(modes):
        mode_vector = eigenvectors[:, i]
        mode_grid = np.zeros((steps, steps))
        mode_grid[indexes] = mode_vector

        eigenmodes[:, :, i] = mode_grid
    
    return eigenvalues, eigenvectors, sorted_eig, eigenmodes


def get_eigenmodes_sparse_square(M, L, dx, modes=6):
    """
    Computes eigenmodes for a square domain using a sparse matrix.

    Args:
        M (np.ndarray): Sparse-compatible matrix to diagonalize.
        L (float): Physical size of the domain.
        dx (float): Grid spacing.
        modes (int): Number of modes to return (default is 6).

    Returns:
        tuple:
            - eigenvalues (np.ndarray): Selected eigenvalues, scaled by 
              dx².
            - eigenvectors (np.ndarray): Corresponding eigenvectors.
            - eigenmodes (np.ndarray): Reshaped eigenmodes of shape 
              (steps, steps, modes).
    """
    steps = int(L/dx)

    M_sparse = scipy.sparse.csr_matrix(M)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(M_sparse, 
        k=modes, which='SM')
    eigenvalues /= dx**2

    eigenmodes = eigenvectors.reshape(steps, steps, -1)

    return eigenvalues, eigenvectors, eigenmodes


def get_eigenmodes_sparse_rectangular(M, L, dx, modes=6):
    """
    Computes eigenmodes for a rectangular domain using a sparse matrix.

    Args:
        M (np.ndarray): Sparse-compatible matrix to diagonalize.
        L (float): Half the width of the rectangular domain.
        dx (float): Grid spacing.
        modes (int): Number of modes to return (default is 6).

    Returns:
        tuple:
            - eigenvalues (np.ndarray): Selected eigenvalues, scaled by dx².
            - eigenvectors (np.ndarray): Corresponding eigenvectors.
            - eigenmodes (np.ndarray): Reshaped modes of shape (steps, 2*steps, modes).
    """
    steps = int(L/dx)

    M_sparse = scipy.sparse.csr_matrix(M)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(M_sparse, 
        k=modes, which='SM')
    eigenvalues /= dx**2

    # Reshape eigenmodes for rectangle (L x 2L)
    eigenmodes = eigenvectors.reshape(steps, 2*steps, -1)

    return eigenvalues, eigenvectors, eigenmodes


def get_eigenmodes_sparse_circular(M, grid, L, dx, modes=6):
    """
    Computes eigenmodes for a circular domain using a sparse matrix.

    Args:
        M (np.ndarray): Sparse-compatible matrix to diagonalize.
        grid (np.ndarray): Boolean mask for circular domain shape.
        L (float): Physical size of the square domain.
        dx (float): Grid spacing.
        modes (int): Number of modes to return (default is 6).

    Returns:
        tuple:
            - eigenvalues (np.ndarray): Selected eigenvalues, scaled by dx².
            - eigenvectors (np.ndarray): Corresponding eigenvectors.
            - eigenmodes (np.ndarray): Modes reshaped to (steps, steps, modes),
              with values only in the circular region.
    """
    steps = int(L/dx)

    M_sparse = scipy.sparse.csr_matrix(M)
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(M_sparse, 
        k=modes, which='SM')
    eigenvalues /= dx**2

    eigenmodes = np.zeros((steps, steps, modes))
    indexes = np.where(grid)

    for i in range(modes):
        mode_vector = eigenvectors[:, i]
        mode_grid = np.zeros((steps, steps))
        mode_grid[indexes] = np.real(mode_vector)
        eigenmodes[:, :, i] = mode_grid

    return eigenvalues, eigenvectors, eigenmodes


def eigenfreqs_size_square(list_L, dx, modes=6):
    """
    Computes eigenfrequencies for square domains of varying size.

    Args:
        list_L (list[float]): List of domain sizes (L) to compute for.
        dx (float): Grid spacing.
        modes (int): Number of eigenfrequencies to compute per domain.

    Returns:
        list[np.ndarray]: List of arrays of eigenfrequencies (per domain 
        size).
    """
    list_eigfreq = []
    for L in list_L:
        steps = int(L/dx)
        
        diag_M = diagonal_matrix(steps)
        eigenvals, _, _ = get_eigenmodes_sparse_square(diag_M, L, dx, modes)
        # Convert to eigenfrequencies
        list_eigfreq.append(np.sqrt(-eigenvals))

    return list_eigfreq
        

def eigenfreqs_size_rectangular(list_L, dx, modes=6):
    """
    Computes eigenfrequencies for rectangular domains (2L wide, L high) 
    of varying size.

    Args:
        list_L (list[float]): List of domain sizes (L) to compute for.
        dx (float): Grid spacing.
        modes (int): Number of eigenfrequencies to compute per domain.

    Returns:
        list[np.ndarray]: List of arrays of eigenfrequencies (per domain 
        size).
    """
    list_eigfreq = []
    for L in list_L:
        steps = int(L/dx)
        
        diag_M = diagonal_matrix(steps)
        eigenvals, _, _ = get_eigenmodes_sparse_rectangular(diag_M, L, dx, modes)
        # Convert to eigenfrequencies
        list_eigfreq.append(np.sqrt(-eigenvals))
    return list_eigfreq


def eigenfreqs_size_circular(list_L, dx, modes=6):
    """
    Computes eigenfrequencies for circular domains of varying size.

    Args:
        list_L (list[float]): List of domain sizes (L) to compute for.
        dx (float): Grid spacing.
        modes (int): Number of eigenfrequencies to compute per domain.

    Returns:
        list[np.ndarray]: List of arrays of eigenfrequencies (per domain 
        size).
    """
    list_eigfreq = []
    for L in list_L:
        steps = int(L/dx)
        
        diag_M = diagonal_matrix(steps)
        grid = circular_domain(steps)
        diag_M = diagonal_circle(grid, diag_M, steps) 

        eigenvals, _, _ = get_eigenmodes_sparse_circular(diag_M, grid, L, 
            dx, modes)
        # Convert to eigenfrequencies
        list_eigfreq.append(np.sqrt(-eigenvals))

    return list_eigfreq


def visualise_eigenfreqs_size(list_L, listlist_eigfreq):
    """
    Plots eigenfrequencies for different domain sizes across three shapes:
    square, rectangular, and circular.

    Args:
        list_L (list[int]): List of domain sizes.
        listlist_eigfreq (list[list[np.ndarray]]): List containing eigenfrequencies
            for each shape (3 total), each as a list over sizes.
    """

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Flatten the axes for correct image rendering
    axes = axes.flatten()
    
    list_L = [[x]*6 for x in list_L]
    domain_shapes = ["Square", "Rectangular", "Circular"]
    for i in range(len(domain_shapes)):
        axes[i].set_axis_on()
        for j in range(len(list_L)):
            axes[i].scatter(list_L[j], listlist_eigfreq[i][j], alpha=0.5)
        axes[i].set_xlim(0,np.max(list_L)+2)
        axes[i].set_xlabel("length")
        axes[i].set_ylabel("eigenfrequencies")
        axes[i].set_title(f"Shape = {domain_shapes[i]}")

    #plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.savefig("results/2Dwave_sizedifferences.pdf")


def time_dependent_visualise_square(eigenmode, eigenfreq, time=1, num_times=4, 
    A=1, B=1, c=1):
    """
    Visualizes the time evolution of a single eigenmode on a square domain.

    Args:
        eigenmode (np.ndarray): 2D eigenmode array.
        eigenfreq (float): Eigenfrequency associated with the mode.
        time (float): Final time for simulation (default is 1).
        num_times (int): Number of time snapshots (default is 4).
        A (float): Amplitude coefficient for cosine term.
        B (float): Amplitude coefficient for sine term.
        c (float): Wave speed (default is 1).
    """
    vmax = np.max(np.abs(eigenmode * (np.cos(np.pi/4) + np.sin(np.pi/4))))

    horizontal = ((num_times // 3) +(num_times % 3))
    fig, axes = plt.subplots(horizontal, 3, figsize=(12, 4*horizontal))
    
    # Flatten the axes for correct image rendering
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()

    for i in range(num_times):
        t = i*(time/(num_times-1))
        u = eigenmode * (A*np.cos(c*eigenfreq*t)+B*np.sin(c*eigenfreq*t))
        axes[i].imshow(u, cmap="bwr", extent=[0, 1, 0, 1], vmin=-vmax, vmax=vmax)
        axes[i].set_axis_on()
        axes[i].set_title(f"t={round(t, 3)}", fontsize = 30)

    plt.tight_layout()
    plt.savefig("results/2Dwave_snapshots.pdf")


def time_dependent_animation(eigenmode, eigenfreq, mode, time=0.01, 
    step=0.0001, A=1, B=1, c=1):
    """
    Creates an animation of the time evolution of a single eigenmode.

    Args:
        eigenmode (np.ndarray): 2D eigenmode array.
        eigenfreq (float): Associated eigenfrequency.
        mode (int): Mode number (used for figure title).
        time (float): Total animation duration (default is 0.01).
        step (float): Time step between animation frames (default is 
            0.0001).
        A (float): Amplitude coefficient for cosine term.
        B (float): Amplitude coefficient for sine term.
        c (float): Wave speed (default is 1).

    Returns:
        matplotlib.animation.ArtistAnimation: Animation of the mode evolution.
    """
    vmax = np.max(np.abs(eigenmode * (np.cos(np.pi/4) + np.sin(np.pi/4))))

    fig, ax = plt.subplots()
    im = ax.imshow(eigenmode, cmap="bwr", extent=[0, 1, 0, 1], vmin=-vmax, 
        vmax=vmax)

    # Store precomputed frames
    ims = []
    for t in np.arange(0+step, time, step):
        u = eigenmode * (A*np.cos(c*eigenfreq*t)+B*np.sin(c*eigenfreq*t))
        im_ = ax.imshow(u, cmap="bwr", extent=[0, 1, 0, 1], animated=True, 
            vmin=-vmax, vmax=vmax)
        ims.append([im_])

    ax.set_title(f"2D Wave Equation (mode = {mode})")
    # Use ArtistAnimation
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    # Close the figure to not display in the notebook
    plt.close(fig)

    return ani


def time_dependent_animation_rectangular(eigenmode, eigenfreq, mode, time=0.01, 
    step=0.0001, A=1, B=1, c=1):
    """
    Creates an animation of the time evolution of a single eigenmode 
    on a rectangular domain (2L wide, L high).

    Args:
        eigenmode (np.ndarray): 2D eigenmode array.
        eigenfreq (float): Associated eigenfrequency.
        mode (int): Mode number (used for figure title).
        time (float): Total animation duration (default is 0.01).
        step (float): Time step between animation frames (default is 0.0001).
        A (float): Amplitude coefficient for cosine term.
        B (float): Amplitude coefficient for sine term.
        c (float): Wave speed (default is 1).

    Returns:
        matplotlib.animation.ArtistAnimation: Animation of the mode evolution.
    """
    vmax = np.max(np.abs(eigenmode * (np.cos(np.pi/4) + np.sin(np.pi/4))))

    fig, ax = plt.subplots()
    im = ax.imshow(eigenmode, cmap="bwr", extent=[0, 2, 0, 1], vmin=-vmax, 
        vmax=vmax, aspect='equal')

    # Store precomputed frames
    ims = []
    for t in np.arange(0+step, time, step):
        u = eigenmode * (A*np.cos(c*eigenfreq*t)+B*np.sin(c*eigenfreq*t))
        im_ = ax.imshow(u, cmap="bwr", extent=[0, 2, 0, 1], animated=True, 
            vmin=-vmax, vmax=vmax, aspect='equal')
        ims.append([im_])

    ax.set_title(f"2D Wave Equation (mode = {mode})")
    # Use ArtistAnimation
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    # Close the figure to not display in the notebook
    plt.close(fig)

    return ani


def visualize_all_eigenfrequencies(L, dx, modes=5):
    """
    Visualizes eigenmodes and corresponding eigenfrequencies for square, 
    rectangular, and circular domains.

    Args:
        L (float): Domain size.
        dx (float): Grid spacing.
        modes (int): Number of modes to visualize per domain 
            (default is 5).

    Returns:
        module: Matplotlib pyplot module (plt) for further use.
    """
    steps = int(L/dx)

    fig, axes = plt.subplots(3, 5, figsize=(12,8))
    axes = axes.flatten()

    # Square grid
    diag_M = diagonal_matrix(steps)
    eigv_sq, _, _, eigm_sq = get_eigenmodes(diag_M, L, dx, modes)
    eigenfre = (-eigv_sq) ** 0.5

    for i in range(modes):
        vmax = np.max(np.abs(eigm_sq[:,:,i]))
        axes[i].imshow(eigm_sq[:, :, i], cmap='bwr', extent=[0, L, 0, L], 
            vmin = -vmax, vmax = vmax)
        axes[i].set_title(f"Mode = {i+1}, λ = {eigenfre[i]:.2f}")

    # Rectangular grid
    diag_M = diagonal_rectangular(steps)

    eigv_rec, _, _, eigm_rec = get_eigenmodes_rectangular(diag_M, L, dx, modes)
    eigenfre = (-eigv_rec) ** 0.5

    for i in range(modes):
        vmax = np.max(np.abs(eigm_rec[:,:,i]))
        im = axes[i+modes].imshow(eigm_rec[:,:,i], cmap='bwr', 
            extent=[0, 2*L, 0, L], vmin=-vmax, vmax=vmax, aspect='equal')
        axes[i+modes].set_title(f"Mode {i+1} λ = {eigenfre[i]:.2f}")

    # Circular grid

    diag_M = diagonal_matrix(steps)
    grid = circular_domain(steps)
    diag_M = diagonal_circle(grid, diag_M, steps) 

    eigv_cir, _, _, eigm_cir = get_eigenmodes_circular(diag_M, grid, L, 
        dx, modes)
    eigenfre = (-eigv_cir) ** 0.5

    for i in range(modes):
        vmax = np.max(np.abs(eigm_cir[:,:,i]))
        axes[i+(2*modes)].imshow(eigm_cir[:, :, i], cmap='bwr', 
            extent=[0, L, 0, L], vmin = -vmax, vmax = vmax)
        axes[i+(2*modes)].set_title(f"Mode = {i+1}, λ = {eigenfre[i]:.2f}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    #plt.tight_layout()
    plt.show()
    fig.savefig("results/eigenfrequencies.pdf", dpi=300)

    return plt


def visualize_diag_matrix(M, N, text='ON'):
    """
    Visualizes a diagonal matrix using a heatmap and optional text 
    labels.

    Args:
        M (np.ndarray): Square matrix of shape (N*N, N*N) to visualize.
        N (int): Grid size used to interpret the shape.
        text (str): If 'ON', display text labels for specific values.

    Returns:
        matplotlib.figure.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.matshow(M, cmap='viridis')


def visualize_all_diagonals(N):
    """
    Visualizes the diagonal matrices for square, rectangular, and circular 
    domains.

    Args:
        N (int): Grid size for each domain.

    Returns:
        module: Matplotlib pyplot module (plt) for further use.
    """
    M_sq = diagonal_matrix(N)

    M_r = diagonal_rectangular(N)

    M_c = diagonal_matrix(N)
    grid = circular_domain(N)
    M_c = diagonal_circle(grid, M_c, N)

    fig, axes = plt.subplots(1, 3, figsize=(9,3))
    axes = axes.flatten()

    axes[0].matshow(M_sq, cmap='viridis')
    axes[0].set_title("Square Domain", fontsize=16)
    axes[1].matshow(M_r, cmap='viridis')
    axes[1].set_title("Rectangular Domain", fontsize=16)
    axes[2].matshow(M_c, cmap='viridis')
    axes[2].set_title("Circular Domain", fontsize=16)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    fig.savefig("results/diagonals.pdf", dpi=300)

    return plt