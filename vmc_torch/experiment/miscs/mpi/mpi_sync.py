from mpi4py import MPI
import numpy as np
import time
import tqdm

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

TARGET_SAMPLE_COUNT = 20  # Target total samples across all ranks
chain_size = TARGET_SAMPLE_COUNT // size

t0 = MPI.Wtime()
local_samples = 0
loc_data_list = []
if rank == 0:
    pbar = tqdm.tqdm(total=TARGET_SAMPLE_COUNT//size)
for _ in range(chain_size):
    loc_data = (rank+0.2)*0.5
    time.sleep(loc_data)  # Simulate some computation time
    buf = (rank, loc_data)
    local_samples += 1
    loc_data_list.append(buf)
    if rank == 0:
        pbar.update(1)

actual_total_samples = COMM.allreduce(local_samples, op=MPI.SUM)

t1 = MPI.Wtime()

print(rank, local_samples, len(loc_data_list), actual_total_samples, t1-t0)