from mpi4py import MPI
import numpy as np
import time
import tqdm

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

for step in range(4):
    TARGET_SAMPLE_COUNT = 20  # Target total samples across all ranks
    terminate = False
    total_samples = 0
    local_samples = 0
    loc_data_list = []
    t0 = MPI.Wtime()
    if rank == 0:
        pbar = tqdm.tqdm(total=TARGET_SAMPLE_COUNT)
        for _ in range(2):
            # just perform 2 iterations on rank 0 so that its loc_data_list is not empty
            loc_data = (rank+0.2)*0.5
            time.sleep(loc_data)  # Simulate some computation time
            buf = (rank, loc_data)
            local_samples += 1
            loc_data_list.append(buf)
            total_samples += 1
            pbar.update(1)
        # Discard messages with tag 1
        while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=2*(step-1)+1):
            redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=2*(step-1)+1)
            print(f"Discarding redundant message from rank {redundant_message[0]}")
        # Then rank 0 monitors the sample generation for all ranks, while itself does not generate samples
        while not terminate:
            # Receive the local sample count from each rank
            buf = COMM.recv(source=MPI.ANY_SOURCE, tag=2*step+1)
            dest_rank = buf[0]
            total_samples += 1
            pbar.update(1)
            # Check if we have enough samples
            if total_samples >= TARGET_SAMPLE_COUNT:
                terminate = True
                for dest_rank in range(1, size):
                    COMM.send(terminate, dest=dest_rank, tag=step+1)
            # Send the termination signal to the rank
            COMM.send(terminate, dest=dest_rank, tag=step+1)
            
        print(rank, total_samples)

        pbar.close()

    else:
        while not terminate:
            loc_data = (rank+0.2)*0.5
            time.sleep(loc_data)  # Simulate some computation time
            buf = (rank, loc_data)
            # Send the local sample count to rank 0
            COMM.send(buf, dest=0, tag=2*step+1)
            # Receive the termination signal from rank 0
            terminate = COMM.recv(source=0, tag=step+1)
            local_samples += 1
            loc_data_list.append(buf)

    actual_total_samples = COMM.allreduce(local_samples, op=MPI.SUM)

    t1 = MPI.Wtime()
        
    print(rank, local_samples, len(loc_data_list), actual_total_samples, t1-t0)
        
