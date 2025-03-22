import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.cm as cm
import typing


def create_grid(N: int, radius: float):
    """Create a square grid that contains the circular disk """
    x = np.linspace(-radius-0.1, radius+0.1, N)
    y = np.linspace(-radius-0.1, radius+0.1, N)
    X, Y = np.meshgrid(x, y)
    h = x[1] - x[0] 
    return X, Y, h


def create_disk_mask(X, Y, radius):
    """Create a boolean mask for points inside the disk """
    return X**2 + Y**2 <= radius**2


def create_index_mapping(mask):
    """Create a mapping from 2D grid indices to 1D system indices """
    N = mask.shape[0]
    map_2d_to_1d = np.zeros((N, N), dtype=int) - 1 
    counter = 0
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                map_2d_to_1d[i, j] = counter
                counter += 1
    return map_2d_to_1d, counter


def find_source_location(x, y, source_x, source_y):
    """Find the grid indices closest to the source coordinates """
    source_i = np.argmin(np.abs(y - source_y))
    source_j = np.argmin(np.abs(x - source_x))
    return source_i, source_j


def adjust_source_location(source_i, source_j, mask, X, Y, source_x, source_y):
    """Adjust source location if it falls outside the domain of simulation """
    if not mask[source_i, source_j]:
        min_dist = float('inf')
        N = mask.shape[0]
        for i in range(N):
            for j in range(N):
                if mask[i, j]:
                    dist = (Y[i, j] - source_y)**2 + (X[i, j] - source_x)**2
                    if dist < min_dist:
                        min_dist = dist
                        source_i, source_j = i, j
        print(f"Source moved to nearest interior point: ({X[source_i, source_j]:.2f}, {Y[source_i, source_j]:.2f})")
    return source_i, source_j


def build_matrix_system(mask, map_2d_to_1d, n_interior, source_i, source_j):
    """Build the coefficient matrix M and right-hand side vector b """
    N = mask.shape[0]
    M = lil_matrix((n_interior, n_interior))
    b = np.zeros(n_interior)
    
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                row = map_2d_to_1d[i, j]
                
                if i == source_i and j == source_j:
                    M[row, row] = 1
                    b[row] = 1
                    continue
                
                M[row, row] = -4  
                
                neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
                for ni, nj in neighbors:
                    if 0 <= ni < N and 0 <= nj < N:
                        if mask[ni, nj]:
                            col = map_2d_to_1d[ni, nj]
                            M[row, col] = 1
    
    return csr_matrix(M), b


def solve_system(M, b):
    """Solve the linear system M . c = b  """
    
    return spsolve(M, b)


def map_solution_to_grid(c, mask, map_2d_to_1d):
    """
    Map the 1D solution vector back to the 2D grid
    """
    N = mask.shape[0]
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                C[i, j] = c[map_2d_to_1d[i, j]]
    return C


def solve_diffusion_direct(N, radius=2.0, source_x=0.6, source_y=1.2):
    """
    Solve the steady-state diffusion equation on a circular disk using direct method.
    """
    X, Y, h = create_grid(N, radius)

    mask = create_disk_mask(X, Y, radius)
    
    map_2d_to_1d, n_interior = create_index_mapping(mask)
    

    x = np.linspace(-radius-0.1, radius+0.1, N)
    y = np.linspace(-radius-0.1, radius+0.1, N)
    source_i, source_j = find_source_location(x, y, source_x, source_y)

    source_i, source_j = adjust_source_location(source_i, source_j, mask, X, Y, source_x, source_y)# If source is not present in the domain 

    M, b = build_matrix_system(mask, map_2d_to_1d, n_interior, source_i, source_j)

    c = solve_system(M, b)

    C = map_solution_to_grid(c, mask, map_2d_to_1d)
    
    return X, Y, C


def plot_disk_boundary(radius):

    """Plotting the boundary of the disk """

    theta = np.linspace(0, 2*np.pi, 100)
    x_boundary = radius * np.cos(theta)
    y_boundary = radius * np.sin(theta)
    plt.plot(x_boundary, y_boundary, 'white', linewidth=2)


def setup_plot_attributes(radius):

    plt.xlim(-radius-0.1, radius+0.1)
    plt.ylim(-radius-0.1, radius+0.1)
    plt.title('Steady-State Concentration on a Circular Disk', fontsize=18)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.grid(False)


def plot_solution(X, Y, C, radius=2.0, source_x=0.6, source_y=1.2):
    """Plot the concentration field of the domain """

    plt.figure(figsize=(10, 8))

    contour = plt.contourf(X, Y, C, levels=40, cmap=cm.magma)
    plt.colorbar(contour, label='Concentration')

    plot_disk_boundary(radius)
    setup_plot_attributes(radius)
    
    plt.tight_layout()
    plt.show()
