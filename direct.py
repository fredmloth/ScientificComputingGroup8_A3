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

N = 100  
radius = 2.0 
source_x = 0.6  
source_y = 1.2 
X, Y, h = create_grid(N, radius)
mask = create_disk_mask(X, Y, radius)



