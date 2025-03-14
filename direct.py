import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
def create_grid(N, radius):
    """Create a square grid that contains the circular disk."""
    x = np.linspace(-radius-0.1, radius+0.1, N)
    y = np.linspace(-radius-0.1, radius+0.1, N)
    X, Y = np.meshgrid(x, y)
    h = x[1] - x[0]  

    return X, Y, h

def create_disk_mask(X, Y, radius):
    """Create a boolean mask for points inside the disk."""
    
    return X**2 + Y**2 <= radius**2


def create_index_mapping(mask):
    """Create a mapping from 2D grid indices to 1D system indices."""

    N = mask.shape[0]
    map_2d_to_1d = np.zeros((N, N), dtype=int) - 1  # We initialize the matrix with -1 and not zero since the indexing starts from zero 
    counter = 0
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                map_2d_to_1d[i, j] = counter
                counter += 1

    return map_2d_to_1d, counter

def find_source_location(x, y, source_x, source_y):
    """Find the grid indices closest to the source coordinates."""
    
    source_i = np.argmin(np.abs(y - source_y))
    source_j = np.argmin(np.abs(x - source_x))
    
    return source_i, source_j

def solve_system(M, b):
    """Solve the linear system M*c = b."""
    return spsolve(M, b)

def map_solution_to_grid(c, mask, map_2d_to_1d):
    """Map the 1D solution vector back to the 2D grid."""
    N = mask.shape[0]
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                C[i, j] = c[map_2d_to_1d[i, j]]
    return C

def build_matrix_system(mask, map_2d_to_1d, n_interior, source_i, source_j):
    """Build the coefficient matrix M and right-hand side vector b."""
    N = mask.shape[0]
    M = lil_matrix((n_interior, n_interior))
    b = np.zeros(n_interior)
    
    for i in range(N):
        for j in range(N):
            if mask[i, j]:
                row = map_2d_to_1d[i, j]
                
                # Check if this is the source point
                if i == source_i and j == source_j:
                    # Source point: set to 1
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

N = 100  
radius = 2.0 
source_x = 0.6  
source_y = 1.2 
X, Y, h = create_grid(N, radius)
mask = create_disk_mask(X, Y, radius)
map_2d_to_1d, n_interior = create_index_mapping(mask)
source_i, source_j = find_source_location(X, Y, source_x, source_y)
M, b = build_matrix_system(mask, map_2d_to_1d, n_interior, source_i, source_j)

c = solve_system(M, b)

# Map solution back to grid
C = map_solution_to_grid(c, mask, map_2d_to_1d)





