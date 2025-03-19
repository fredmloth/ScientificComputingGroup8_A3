# doesn't really work (sorta does?) and i have no idea if im going in the right direction at all

import matplotlib.pyplot as plt
import numpy as np
import diagonal_matrix as dm

A = B = 1
c = 1
modes = 1
length = 50

diag_M = dm.diagonal_matrix(length)

eigenval, _, _, eigenmode = dm.get_eigenmodes(diag_M, length, modes)

for t in np.arange(0, 10, 0.01):
    u = eigenmode * (A * np.cos(c*eigenval*t) + B * np.sin(c*eigenval*t))
    plt.imshow(u, cmap='bwr', extent=[0, 1, 0, 1])
    plt.draw()
    plt.pause(0.1)
    plt.clf()