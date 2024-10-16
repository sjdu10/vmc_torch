from mpi4py import MPI
import mpi4py
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

print(SIZE, RANK)