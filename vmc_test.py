import torch
import numpy as np
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

# Root rank 0 creates the data to broadcast, others create an empty array
if RANK == 0:
    test_vec = np.array([1,2,3,4], dtype=np.float32)  # Convert to NumPy for MPI
else:
    # Create an empty NumPy array with the same dtype as on rank 0
    test_vec = np.zeros(4, dtype=np.float32)  # Ensure correct dtype and initialization

# Ensure the array is contiguous before broadcasting
test_vec = np.ascontiguousarray(test_vec)

# All ranks participate in the broadcast
COMM.Bcast(test_vec, root=0)


print(f"Rank {RANK} received new_param_vec: {test_vec}")
