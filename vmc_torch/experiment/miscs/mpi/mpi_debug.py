from mpi4py import MPI
import numpy as np
from autoray import do
import scipy.sparse.linalg as spla

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main_sampling():

    # Simple check for at least 2 processes
    if size < 2:
        if rank == 0:
            print("This program requires at least 2 MPI processes")
        return

    # Tag for message identification
    TAG = 42
    messages = []
    Ns = 10*size

    terminate = np.array([0], dtype=np.int32)

    if rank != 0:
        while not terminate[0]:
            # Assume some Markov Chain computation is done here
            ...

            # Get terminiation signal from rank 0
            send_rank = np.array([rank], dtype=np.int32)
            comm.Send([send_rank, MPI.INT], dest=0, tag=TAG)
            terminate = np.empty(1, dtype=np.int32)
            comm.Recv([terminate, MPI.INT], source=0, tag=TAG+1)

    elif rank == 0:
        while not terminate[0]:
            # Send termination signal to all other ranks
            recv_rank = np.empty(1, dtype=np.int32)
            comm.Recv([recv_rank, MPI.INT], source=MPI.ANY_SOURCE, tag=TAG)
            messages.append(recv_rank[0])
            if len(messages) >= Ns:
                terminate[0] = 1
                # Termination, send signal 1 to all other ranks
                for i in range(1, size):
                    comm.Send([terminate, MPI.INT], dest=i, tag=TAG+1)
            else:
                # No termination, send signal 0 back to the sender
                comm.Send([terminate, MPI.INT], dest=recv_rank[0], tag=TAG+1)

    # Print the messages on each rank, should be empty except for rank 0
    # For rank 0 the length of the messages should be equal to Ns
    print(f"Rank {rank} messages: {messages}, length: {len(messages)}")

    # Synchronize all processes
    comm.Barrier()

def main_SR():

    n_local_samples = 50
    gradient_size = 10000
    local_logamp_grad_matrix = np.random.rand(gradient_size, n_local_samples)
    local_mean_logamp_grad = np.mean(local_logamp_grad_matrix, axis=1)
    mean_logamp_grad = comm.allreduce(local_mean_logamp_grad, op=MPI.SUM)/size
    energy_grad = np.random.rand(gradient_size)
    if rank != 0:
        energy_grad = None

    energy_grad = comm.bcast(energy_grad, root=0)

    total_samples = comm.allreduce(n_local_samples, op=MPI.SUM)

    if rank == 0:
        print("Total sample size:", total_samples)
        print("Matrix A shape:", f'{len(mean_logamp_grad)} x {len(mean_logamp_grad)}')
        print("Solve Ax=b via iterative solver which uses MPI communication to compute matrix-vector product...")

    if energy_grad is None:
        raise ValueError("energy_grad is None")

    # print("energy_grad", energy_grad, RANK)
    def R_dot_x(x, eta=1e-6):
        x_out_local = np.zeros_like(x)
        # use matrix multiplication for speedup
        x_out_local = do('dot', local_logamp_grad_matrix, do('dot', local_logamp_grad_matrix.T, x))
        # synchronize the result
        x_out = comm.allreduce(x_out_local, op=MPI.SUM)/total_samples
        x_out -= np.dot(mean_logamp_grad, x)*mean_logamp_grad
        return x_out + eta*x

    n = gradient_size
    matvec = lambda x: R_dot_x(x, 1e-3)
    A = spla.LinearOperator((n, n), matvec=matvec)
    b = energy_grad

    t0 = MPI.Wtime()
    solver = spla.minres
    dp, info = solver(A, b, maxiter=1000, rtol=1e-4)
    t1 = MPI.Wtime()
    if rank == 0:
        print("    Time for solving the linear equation: ", t1-t0)
        print("    SR solver convergence? ", 'Yes' if info == 0 else 'No')

if __name__ == "__main__":
    if rank == 0:
        print("Starting sampling test...")
    main_sampling()
    comm.Barrier()
    if rank == 0:
        print("\nStarting SR test...")
    main_SR()