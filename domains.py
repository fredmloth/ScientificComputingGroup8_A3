"""
This programme creates a grid of which points on the boundary and outside of
the domain are set to zero; other points are set to one.

Domain options:
- square_domain: for a square domain of LxL
- rectangular_domain: for a rectangular domain or Lx2L
- circular_domain: for a circular domain with diameter L
"""

import numpy as np
import matplotlib.pyplot as plt

def square_domain(L, steps):
    grid: np.array[np.array[float]] = np.zeros((steps+1, steps+1))

    grid[1:-1,1:-1] = 1

    # plots domain with boundary conditions set
    plt.imshow(grid, cmap="plasma", extent=[0,L,0,L])
    # plt.show()

    return grid

###############################################################################

def rectangular_domain(L, steps):
    grid: np.array[np.array[float]] = np.zeros((steps+1, (2*steps)+1))
    
    grid[1:-1,1:-1] = 1

    # plots domain with boundary conditions set
    plt.imshow(grid, cmap="plasma", extent=[0,2*L,0,L])
    # plt.show()

    return grid

###############################################################################

def get_neighbours(point, steps):
    i, j = point
    neighbours = []

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for direction in directions:
        if 0 <= i+direction[0] <= steps and 0 <= j+direction[1] <= steps:
            neighbours.append((i+direction[0], j+direction[1]))

    return neighbours

# to make the values outside of the circle all zero at once
def circular_domain(L, steps):
    """
    Creates a grid where all points outside of the circular domain and points
    on the boundary have value zero, and points inside the domain have value 1.

    Parameters:
    - L (float): the diameter of the circular domain.
    - steps (int): the number of steps to divide the domain into.
    """
    grid: np.array[np.array[float]] = np.ones((steps+1, steps+1))
    r: float = L/2
    coordinates_outside_domain: list[list[int]] = []
    x: np.array[float] = np.linspace(0, L, steps+1)
    y: np.array[float] = np.linspace(0, L, steps+1)

    # sets all points outside of the circular domain to zero and saves their
    # coordinates
    for i in y:
        for j in x:
            if np.sqrt(((j-r) ** 2 + (i-r) ** 2)) > r:
                coordinates_outside_domain.append((round(i*steps),round(j*steps)))
                grid[round(i*steps),round(j*steps)] = 0

    # sets all points on teh boundary of the circular domain to zero
    for i in y:
        for j in x:
            point = (round(i*steps), round(j*steps))
            for neighbour in get_neighbours(point, steps):
                if neighbour in coordinates_outside_domain and grid[point] == 1:
                    grid[point] = 0

    # plots domain with boundary conditions set
    plt.imshow(grid, cmap="plasma", extent=[0,L,0,L])
    # plt.show()

    return grid

###############################################################################

L = 1

# must be even for square domain
steps = 8

# square_domain(L, steps)
# rectangular_domain(L, steps)
# circular_domain(L, steps)