import numpy as np
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

N = 100  
radius = 2.0 
source_x = 0.6  
source_y = 1.2 
X, Y, h = create_grid(N, radius)
mask = create_disk_mask(X, Y, radius)
map_2d_to_1d, n_interior = create_index_mapping(mask)



