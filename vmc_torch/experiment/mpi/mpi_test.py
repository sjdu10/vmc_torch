def mpi_check():
    """
    Check if mpi4py is installed and working correctly.
    """
    try:
        import mpi4py
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print(f"Rank {rank} out of {size} ranks.")
        return True
    except ImportError:
        print("mpi4py is not installed.")
        return False
    
if __name__ == "__main__":
    mpi_check()