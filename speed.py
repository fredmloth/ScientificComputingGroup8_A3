"""
This programme measures the difference in time between
scipy.linalg.eigs() and scipy.sparse.linalg.eigs().

DOES NOT WORK AT ALL YET GOTTA RESEARCH MORE
"""

from scipy import linalg, sparse
import time
from diagonal_matrix import diagonal_matrix

for length in range(50, 200, 10):
    start_time = time.time()
    eigval, eigvec = linalg.eigh(diagonal_matrix(length))
    end_time = time.time()

    time_taken_linalg = end_time - start_time

    print(time_taken_linalg)

    start_time = time.time()
    eigval, eigvec = sparse.linalg.eigs(diagonal_matrix(length))
    end_time = time.time()

    time_taken_sparse = end_time - start_time

    print(time_taken_sparse)

    print(time_taken_linalg, time_taken_sparse)