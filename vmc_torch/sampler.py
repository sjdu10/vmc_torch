import os
import numpy as np
from mpi4py import MPI
import random
from tqdm import tqdm
import time
import pyinstrument

# torch
import torch

# quimb
from autoray import do

#jax
import jax
import random

from .global_var import DEBUG, set_debug, TIME_PROFILING, TAG_OFFSET

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# --- Utils ---
def from_netket_config_to_quimb_config(netket_configs):
    def func(netket_config):
        """Translate netket spin-1/2 config to tensor network product state config"""
        total_sites = len(netket_config)//2
        spin_up = netket_config[:total_sites]
        spin_down = netket_config[total_sites:]
        sum_spin = spin_up + spin_down
        quimb_config = np.zeros(total_sites, dtype=int)
        for i in range(total_sites):
            if sum_spin[i] == 0:
                quimb_config[i] = 0
            if sum_spin[i] == 2:
                quimb_config[i] = 3
            if sum_spin[i] == 1:
                if spin_down[i] == 1:
                    quimb_config[i] = 1
                else:
                    quimb_config[i] = 2
        return quimb_config
    if len(netket_configs.shape) == 1:
        return func(netket_configs)
    else:
        # batched
        return np.array([func(netket_config) for netket_config in netket_configs])

def from_quimb_config_to_netket_config(quimb_config):
    """Translate tensor network product state config to netket spin-1/2 config"""
    total_sites = len(quimb_config)
    spin_up = np.zeros(total_sites, dtype=int)
    spin_down = np.zeros(total_sites, dtype=int)
    for i in range(total_sites):
        if quimb_config[i] == 0:
            spin_up[i] = 0
            spin_down[i] = 0
        if quimb_config[i] == 1:
            spin_up[i] = 0
            spin_down[i] = 1
        if quimb_config[i] == 2:
            spin_up[i] = 1
            spin_down[i] = 0
        if quimb_config[i] == 3:
            spin_up[i] = 1
            spin_down[i] = 1
    return np.concatenate((spin_up, spin_down))

# --- Sampler ---

class Sampler:
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100, reset_chain=False, equal_partition=True, dtype=torch.float32):
        self.hi = hi
        self.Ns = N_samples
        self.graph = graph
        self.burn_in_steps = burn_in_steps
        self.reset_chain = reset_chain
        self.dtype = dtype
        self.initial_config = None
        self.current_config = None
        self.equal_partition = equal_partition
        self.reset()
    
    def reset(self):
        """Reset the current sampler configuration to a random config in the Hilbert space."""
        rand_int = random.randint(0, 2**32-1)
        self.initial_config = torch.tensor(np.asarray(self.hi.random_state(jax.random.PRNGKey(rand_int))), dtype=self.dtype)
        self.current_config = self.initial_config.clone()

    def _sample_next(self, vstate_func):
        """Sample the next configuration."""
        raise NotImplementedError
    
    def sample(self, vstate, op, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration."""
        raise NotImplementedError
    

class MetropolisExchangeSampler(Sampler):
    """Metropolis-Hastings sampler that uses the exchange move as the sampling rule."""
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100, reset_chain=False, random_edge=False, subchain_length=None, equal_partition=True, dtype=torch.float32):
        super().__init__(hi, graph, N_samples, burn_in_steps, reset_chain, equal_partition)
        self.random_edge = random_edge
        self.subchain_length = graph.n_edges if subchain_length is None else subchain_length
    
    def burn_in(self, vstate):
        """Discard the initial samples. (Burn-in)"""
        for _ in range(self.burn_in_steps):
            self._sample_next(vstate)
    
    def _sample_next(self, vstate):
        """Sample the next configuration. Change the current configuration in place.
        Must be implemented in the derived class."""
        raise NotImplementedError

    def sample(self, vstate, op, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration.
        return a tuple of (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat)"""
        assert self.equal_partition, "The number of samples must be equal for all MPI processes."
        if RANK == 0:
            print('Burn-in...')
            t_burnin0 = time.time()
        self.burn_in(vstate)
        if RANK == 0:
            t_burnin1 = time.time()
            print('Burn-in time:', t_burnin1 - t_burnin0)

        op_loc_sum = 0

        # We use Welford's Algorithm to compute the sample variance of op_loc in a single pass.
        n = 0
        op_loc_mean = 0
        op_loc_M2 = 0
        op_loc_var = 0
        op_loc_vec = np.zeros(chain_length)

        if RANK == 0:
            pbar = tqdm(total=chain_length, desc='Sampling starts for rank 0...')
        
        for chain_step in range(chain_length):
            time0 = MPI.Wtime()
            # sample the next configuration
            psi_sigma = 0
            while psi_sigma == 0:
                # We need to make sure that the amplitude is not zero
                sigma, psi_sigma = self._sample_next(vstate)
                    
            n += 1
        
            # compute local energy and amplitude gradient
            psi_sigma = vstate.amplitude(sigma)
            # compute the connected non-zero operator matrix elements <eta|O|sigma>
            eta, O_etasigma = op.get_conn(sigma) # Non-zero matrix elements and corresponding configurations
            psi_eta = vstate.amplitude(eta)

            # convert torch tensors to numpy arrays
            psi_sigma = psi_sigma.cpu().detach().numpy()
            psi_eta = psi_eta.cpu().detach().numpy()

            # compute the local operator
            op_loc = np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)
            op_loc_vec[chain_step] = op_loc

            # accumulate the local operator
            op_loc_sum += op_loc

            # update the sample variance of op_loc
            op_loc_mean_prev = op_loc_mean
            op_loc_mean += (op_loc - op_loc_mean) / n
            op_loc_M2 += (op_loc - op_loc_mean_prev) * (op_loc - op_loc_mean)

            time1 = MPI.Wtime()
            if RANK == 0:
                pbar.set_postfix({'Time per sample for each rank': (time1 - time0)})

            # update the sample variance
            if n > 1:
                op_loc_var = op_loc_M2 / (n - 1)

            # total_n = COMM.reduce(n, op=MPI.SUM, root=0)
            # add a progress bar if rank == 0
            if RANK == 0:
                pbar.update(1)
        
        
        if self.reset_chain:
            self.reset()

        if RANK == 0:
            pbar.close()
        
        # The following is for computing the Rhat diagnostic using the Gelman-Rubin formula

        # Step 1：split the chain in half and compute the within chain variance
        split_chains = np.split(op_loc_vec, 2)
        # Step 2: compute the within chain variance
        W_loc = np.sum([np.var(split_chain) for split_chain in split_chains])
        chain_means_loc = [np.mean(split_chain) for split_chain in split_chains]

        samples = (op_loc_sum, op_loc_var, W_loc, chain_means_loc)
        return samples
    
    def sample_eager(self, vstate, op, message_tag=None):
        """Sample eagerly for the local energy and amplitude gradient for each configuration.
        return a tuple of (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat)"""
        assert self.equal_partition == False, 'Must not use equal partition for eager sampling.'

        if RANK == 0:
            print('Burn-in...')
            t_burnin0 = time.time()
        self.burn_in(vstate)
        if RANK == 0:
            t_burnin1 = time.time()
            print('Burn-in time:', t_burnin1 - t_burnin0)

        op_loc_sum = 0

        # We only estimate the variance of op_loc locally
        n = 0
        n_total = 0
        op_loc_vec = []
        terminate = False

        if RANK == 0:
            pbar = tqdm(total=self.Ns, desc='Sampling starts...')
            for _ in range(2):
                op_loc, _ = self._sample_expect(vstate, op)
                op_loc_vec.append(op_loc)
                op_loc_sum += op_loc

                n += 1
                n_total += 1
                pbar.update(1)
            
            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1):
                redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1)
                del redundant_message
            
            while not terminate:
                # Receive the local sample count from each rank
                buf = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET)
                dest_rank = buf[0]
                n_total += 1
                pbar.update(1)
                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = True
                    for dest_rank in range(1, SIZE):
                        COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                # Send the termination signal to the rank
                COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
            
            pbar.close()
        
        else:
            while not terminate:
                op_loc, _ = self._sample_expect(vstate, op)
                n += 1
                op_loc_vec.append(op_loc)
                # accumulate the local energy and amplitude gradient
                op_loc_sum += op_loc

                buf = (RANK,)
                # Send the local sample count to rank 0
                COMM.send(buf, dest=0, tag=message_tag+TAG_OFFSET)
                # Receive the termination signal from rank 0
                terminate = COMM.recv(source=0, tag=message_tag+1)

        
        if self.reset_chain:
            self.reset()
        
        if RANK == 0:
            pbar.close()
        
        # convert the list to numpy array
        op_loc_vec = np.asarray(op_loc_vec)
        if n > 1:
            op_loc_var = np.var(op_loc_vec, ddof=1)
        else:
            op_loc_var = 0
        samples = (op_loc_sum, op_loc_var, n)

        return samples

    def sample_w_grad(self, vstate, op, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration.
        return a tuple of (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat)"""
        assert self.equal_partition, "The number of samples must be equal for all MPI processes."
        if RANK == 0:
            print('Burn-in...')
            t_burnin0 = time.time()
        self.burn_in(vstate)
        if RANK == 0:
            t_burnin1 = time.time()
            print('Burn-in time:', t_burnin1 - t_burnin0)

        op_loc_sum = 0
        logpsi_sigma_grad_sum = np.zeros(vstate.Np)
        op_logpsi_sigma_grad_product_sum = np.zeros(vstate.Np)

        logpsi_sigma_grad_mat = np.zeros((vstate.Np, chain_length))

        # We use Welford's Algorithm to compute the sample variance of op_loc in a single pass.
        n = 0
        op_loc_mean = 0
        op_loc_M2 = 0
        op_loc_var = 0
        op_loc_vec = np.zeros(chain_length)

        if RANK == 0:
            pbar = tqdm(total=chain_length, desc='Sampling starts for rank 0...')
        
        for chain_step in range(chain_length):
            time0 = MPI.Wtime()
            # sample the next configuration
            psi_sigma = 0
            while psi_sigma == 0:
                # We need to make sure that the amplitude is not zero
                sigma, psi_sigma = self._sample_next(vstate)

            time1 = MPI.Wtime()
            n += 1
            if RANK == 0:
                pbar.set_postfix({'Time per sample for each rank': (time1 - time0)})

            # compute local energy and amplitude gradient
            psi_sigma, logpsi_sigma_grad = vstate.amplitude_grad(sigma)
            # compute the connected non-zero operator matrix elements <eta|O|sigma>
            eta, O_etasigma = op.get_conn(sigma) # Non-zero matrix elements and corresponding configurations
            psi_eta = vstate.amplitude(eta)

            # convert torch tensors to numpy arrays
            psi_sigma = psi_sigma.cpu().detach().numpy()
            psi_eta = psi_eta.cpu().detach().numpy()
            logpsi_sigma_grad = logpsi_sigma_grad.cpu().detach().numpy()

            # compute the local operator
            op_loc = np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)
            op_loc_vec[chain_step] = op_loc

            # accumulate the local energy and amplitude gradient
            op_loc_sum += op_loc
            logpsi_sigma_grad_sum += logpsi_sigma_grad
            op_logpsi_sigma_grad_product_sum += op_loc * logpsi_sigma_grad

            # collect the log-amplitude gradient
            logpsi_sigma_grad_mat[:, chain_step] = logpsi_sigma_grad

            # update the sample variance of op_loc
            op_loc_mean_prev = op_loc_mean
            op_loc_mean += (op_loc - op_loc_mean) / n
            op_loc_M2 += (op_loc - op_loc_mean_prev) * (op_loc - op_loc_mean)

            # update the sample variance
            if n > 1:
                op_loc_var = op_loc_M2 / (n - 1)

            # total_n = COMM.reduce(n, op=MPI.SUM, root=0)
            # add a progress bar if rank == 0
            if RANK == 0:
                pbar.update(1)
        
        
        if self.reset_chain:
            self.reset()

        if RANK == 0:
            pbar.close()
        
        # The following is for computing the Rhat diagnostic using the Gelman-Rubin formula

        # Step 1：split the chain in half and compute the within chain variance
        split_chains = np.split(op_loc_vec, 2)
        # Step 2: compute the within chain variance
        W_loc = np.sum([np.var(split_chain) for split_chain in split_chains])
        chain_means_loc = [np.mean(split_chain) for split_chain in split_chains]

        samples = (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat, W_loc, chain_means_loc)
        return samples
    
    def sample_w_grad_eager(self, vstate, op, message_tag=None):
        """Sample eagerly for the local energy and amplitude gradient for each configuration.
        return a tuple of (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat)"""
        assert self.equal_partition == False, 'Must not use equal partition for eager sampling.'

        if RANK == 0:
            print('Burn-in...')
            t_burnin0 = time.time()
        self.burn_in(vstate)
        if RANK == 0:
            t_burnin1 = time.time()
            print('Burn-in time:', t_burnin1 - t_burnin0)

        op_loc_sum = 0
        logpsi_sigma_grad_sum = np.zeros(vstate.Np)
        op_logpsi_sigma_grad_product_sum = np.zeros(vstate.Np)

        logpsi_sigma_grad_mat = []
        # We only estimate the variance of op_loc locally
        n = 0
        n_total = 0
        op_loc_vec = []
        terminate = False

        if RANK == 0:
            pbar = tqdm(total=self.Ns, desc='Sampling starts...')
            for _ in range(2):
                op_loc, logpsi_sigma_grad, _ = self._sample_expect_grad(vstate, op)
                op_loc_vec.append(op_loc)
                op_loc_sum += op_loc
                logpsi_sigma_grad_sum += logpsi_sigma_grad # summed to save memory ?
                #XXX: summing is not necessary, we can put all post-processing of the samples in variational_state module
                op_logpsi_sigma_grad_product_sum += op_loc * logpsi_sigma_grad
                logpsi_sigma_grad_mat.append(logpsi_sigma_grad)

                n += 1
                n_total += 1
                pbar.update(1)
            
            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1):
                redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1)
                del redundant_message
            
            while not terminate:
                # Receive the local sample count from each rank
                buf = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET)
                dest_rank = buf[0]
                n_total += 1
                pbar.update(1)
                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = True
                    for dest_rank in range(1, SIZE):
                        COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                # Send the termination signal to the rank
                COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
            
            pbar.close()
        
        else:
            while not terminate:
                op_loc, logpsi_sigma_grad, _ = self._sample_expect_grad(vstate, op)
                n += 1
                op_loc_vec.append(op_loc)
                # accumulate the local energy and amplitude gradient
                op_loc_sum += op_loc
                logpsi_sigma_grad_sum += logpsi_sigma_grad
                op_logpsi_sigma_grad_product_sum += op_loc * logpsi_sigma_grad

                # collect the log-amplitude gradient
                logpsi_sigma_grad_mat.append(logpsi_sigma_grad)

                buf = (RANK,)
                # Send the local sample count to rank 0
                COMM.send(buf, dest=0, tag=message_tag+TAG_OFFSET)
                # Receive the termination signal from rank 0
                terminate = COMM.recv(source=0, tag=message_tag+1)

        
        if self.reset_chain:
            self.reset()
        
        if RANK == 0:
            pbar.close()
        
        # convert the list to numpy array
        logpsi_sigma_grad_mat = np.asarray(logpsi_sigma_grad_mat).T
        op_loc_vec = np.asarray(op_loc_vec)
        if n > 1:
            op_loc_var = np.var(op_loc_vec, ddof=1)
        else:
            op_loc_var = 0
        samples = (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat)

        return samples

    def _sample_expect_grad(self, vstate, op):
        """Get one sample of the local energy and amplitude gradient."""
        time0 = MPI.Wtime()
        # sample the next configuration
        psi_sigma = 0
        while psi_sigma == 0:
            # We need to make sure that the amplitude is not zero
            sigma, psi_sigma = self._sample_next(vstate)
        time1 = MPI.Wtime()

        # compute local energy and amplitude gradient
        psi_sigma, logpsi_sigma_grad = vstate.amplitude_grad(sigma)
        # compute the connected non-zero operator matrix elements <eta|O|sigma>
        eta, O_etasigma = op.get_conn(sigma)
        psi_eta = vstate.amplitude(eta)

        # convert torch tensors to numpy arrays
        psi_sigma = psi_sigma.cpu().detach().numpy()
        psi_eta = psi_eta.cpu().detach().numpy()
        logpsi_sigma_grad = logpsi_sigma_grad.cpu().detach().numpy()

        # compute the local operator
        op_loc = np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)

        return op_loc, logpsi_sigma_grad, time1 - time0
    
    def _sample_expect(self, vstate, op):
        """Get one sample of the local operator."""
        time0 = MPI.Wtime()
        # sample the next configuration
        psi_sigma = 0
        while psi_sigma == 0:
            # We need to make sure that the amplitude is not zero
            sigma, psi_sigma = self._sample_next(vstate)
        time1 = MPI.Wtime()

        # compute local energy and amplitude gradient
        psi_sigma = vstate.amplitude(sigma)
        # compute the connected non-zero operator matrix elements <eta|O|sigma>
        eta, O_etasigma = op.get_conn(sigma)
        psi_eta = vstate.amplitude(eta)

        # convert torch tensors to numpy arrays
        psi_sigma = psi_sigma.cpu().detach().numpy()
        psi_eta = psi_eta.cpu().detach().numpy()

        # compute the local operator
        op_loc = np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)

        return op_loc, time1 - time0
    
    def sample_expectation(self, vstate, op, chain_length=1):
        """Sample the expectation value of the operator `op`."""
        self.burn_in(vstate)
        E_loc_list = []
        configs = []
        op_loc_sum = 0
        for chain_step in range(chain_length):
            sigma, psi_sigma = self._sample_next(vstate)
            psi_sigma = vstate.amplitude(sigma)
            eta, O_etasigma = op.get_conn(sigma)
            psi_eta = vstate.amplitude(eta)
            psi_sigma = psi_sigma.cpu().detach().numpy()
            psi_eta = psi_eta.cpu().detach().numpy()
            op_loc = O_etasigma @ (psi_eta / psi_sigma)
            op_loc_sum += op_loc
            E_loc_list.append(op_loc)
            configs.append(sigma)
        return op_loc_sum / chain_length, E_loc_list, configs

    def sample_dense(self, vstate, op):
        all_config = self.hi.all_states()
        all_config = np.asarray(all_config)
        psi_vec = vstate.amplitude(all_config)
        op_dense = op.to_dense()
        expect_op = psi_vec.conj().T @ (op_dense @ psi_vec)/(psi_vec.conj().T @ psi_vec)
        return expect_op
    
    def sample_SWO_dataset_eager(self, vstate, op, message_tag=None):
        """
        Sample the configurations {c}_(t-1) according to |<c|Psi_(t-1)>|^2, 
        and collect the corresponding amplitudes <c|Psi_(t-1)>, <c|op|Psi_(t-1)> for supervised wavefunction optimization in each rank.

        Parameters
        ----------
        vstate : VariationalState
            The variational state.
        op : CustomizedOperator
            The operator.
        message_tag : int
            The message tag for MPI communication.

        Returns
        -------
        list
            A list of tuples (config, <c|Psi_(t-1)>, <c|op|Psi_(t-1)>).
        """
        if RANK == 0:
            print('Burn-in...')
            t_burnin0 = time.time()
        self.burn_in(vstate)
        if RANK == 0:
            t_burnin1 = time.time()
            print('Burn-in time:', t_burnin1 - t_burnin0)

        n = 0
        n_total = 0
        config_list = []
        config_amplitudes_dict = {}
        terminate = False

        if RANK == 0:
            pbar = tqdm(total=self.Ns, desc='Sampling starts...')
            for _ in range(2):
                config, psi_sigma = self._sample_next(vstate)
                config_list.append(config)
                if config not in config_amplitudes_dict:
                    eta, O_etasigma = op.get_conn(config)
                    psi_eta = vstate.amplitude(eta)
                    psi_eta = psi_eta.cpu().detach().numpy()
                    psi_sigma = psi_sigma.cpu().detach().numpy()
                    op_psi_eta = np.sum(O_etasigma * psi_eta, axis=-1)
                    config_amplitudes_dict[config] = (psi_sigma, op_psi_eta)
                else:
                    psi_sigma, op_psi_eta = config_amplitudes_dict[config] #XXX: not needed actually
                n += 1
                n_total += 1
                pbar.update(1)
            
            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1):
                redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1)
                del redundant_message
            
            while not terminate:
                # Receive the local sample count from each rank
                buf = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET)
                dest_rank = buf[0]
                n_total += 1
                pbar.update(1)
                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = True
                    for dest_rank in range(1, SIZE):
                        COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                # Send the termination signal to the rank
                COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
            
            pbar.close()
        
        else:
            while not terminate:
                config, psi_sigma = self._sample_next(vstate)
                config_list.append(config)
                if config not in config_amplitudes_dict:
                    eta, O_etasigma = op.get_conn(config)
                    psi_eta = vstate.amplitude(eta)
                    psi_eta = psi_eta.cpu().detach().numpy()
                    psi_sigma = psi_sigma.cpu().detach().numpy()
                    op_psi_eta = np.sum(O_etasigma * psi_eta, axis=-1)
                    config_amplitudes_dict[config] = (psi_sigma, op_psi_eta)
                else:
                    psi_sigma, op_psi_eta = config_amplitudes_dict[config]
                n += 1

                buf = (RANK,)
                # Send the local sample count to rank 0
                COMM.send(buf, dest=0, tag=message_tag+TAG_OFFSET)
                # Receive the termination signal from rank 0
                terminate = COMM.recv(source=0, tag=message_tag+1)
            
        if self.reset_chain:
            self.reset()

        if RANK == 0:
            pbar.close()
        
        dataset = (config_list, config_amplitudes_dict)

        return dataset
    
    def sample_SWO_state_fitting_dataset_eager(self, vstate, target_state, message_tag=None, compute_energy=False, op=None):
        """
        Sample the configurations {c}_(t-1) according to |<c|Psi_(t-1)>|^2, 
        and collect the corresponding amplitudes <c|Psi_(t-1)>, <c|Psi_target> for supervised wavefunction fitting in each rank.

        Parameters
        ----------
        vstate : VariationalState
            The variational state.
        target_state : VariationalState
            The target state.
        message_tag : int
            The message tag for MPI communication.

        Returns
        -------
        list
            A list of tuples (config, <c|Psi_(t-1)>, <c|Psi_target>).
        """
        if RANK == 0:
            print('Burn-in...')
            t_burnin0 = time.time()
        self.burn_in(vstate)
        if RANK == 0:
            t_burnin1 = time.time()
            print('Burn-in time:', t_burnin1 - t_burnin0)

        n = 0
        n_total = 0
        config_list = []
        config_amplitudes_dict = {}
        terminate = False
        op_expect = 0

        if RANK == 0:
            pbar = tqdm(total=self.Ns, desc='Sampling starts...')
            for _ in range(2):
                config, psi_sigma = self._sample_next(vstate)
                config_list.append(config)
                if config not in config_amplitudes_dict:
                    psi_target_sigma = target_state.amplitude(config)
                    config_amplitudes_dict[config] = (psi_sigma, psi_target_sigma)
                else:
                    psi_sigma, psi_target_sigma = config_amplitudes_dict[config] #XXX: not needed actually
                if compute_energy:
                    eta, O_etasigma = op.get_conn(config)
                    psi_eta = vstate.amplitude(eta)
                    psi_eta = psi_eta.cpu().detach().numpy()
                    psi_sigma = psi_sigma.cpu().detach().numpy()
                    op_loc = np.sum(O_etasigma * psi_eta / psi_sigma, axis=-1)
                    op_expect += op_loc
                    ...#XXX: compute variance

                n += 1
                n_total += 1
                pbar.update(1)
            
            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1):
                redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1)
                del redundant_message
            
            while not terminate:
                # Receive the local sample count from each rank
                buf = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET)
                dest_rank = buf[0]
                n_total += 1
                pbar.update(1)
                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = True
                    for dest_rank in range(1, SIZE):
                        COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                # Send the termination signal to the rank
                COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
            
            pbar.close()
        
        else:
            while not terminate:
                config, psi_sigma = self._sample_next(vstate)
                config_list.append(config)
                if config not in config_amplitudes_dict:
                    psi_target_sigma = target_state.amplitude(config)
                    config_amplitudes_dict[config] = (psi_sigma, psi_target_sigma)
                else:
                    psi_sigma, psi_target_sigma = config_amplitudes_dict[config]
                if compute_energy:
                    eta, O_etasigma = op.get_conn(config)
                    psi_eta = vstate.amplitude(eta)
                    psi_eta = psi_eta.cpu().detach().numpy()
                    psi_sigma = psi_sigma.cpu().detach().numpy()
                    op_loc = np.sum(O_etasigma * psi_eta / psi_sigma, axis=-1)
                    op_expect += op_loc
                    ...#XXX: compute variance
                n += 1

                buf = (RANK,)
                # Send the local sample count to rank 0
                COMM.send(buf, dest=0, tag=message_tag+TAG_OFFSET)
                # Receive the termination signal from rank 0
                terminate = COMM.recv(source=0, tag=message_tag+1)
            
        if self.reset_chain:
            self.reset()

        if RANK == 0:
            pbar.close()
        
        dataset = (config_list, config_amplitudes_dict, op_expect)

        return dataset

        


class MetropolisExchangeSamplerSpinless(MetropolisExchangeSampler):
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100, reset_chain=False, random_edge=False, subchain_length=None, equal_partition=True, dtype=torch.float32):
        super().__init__(hi, graph, N_samples, burn_in_steps, reset_chain, random_edge, subchain_length, equal_partition, dtype)
    
    def _sample_next(self, vstate):
        """Sample the next configuration. Change the current configuration in place."""
        current_amp = vstate.amplitude(self.current_config)
        current_prob = abs(current_amp)**2
        proposed_config = self.current_config.clone()
        if self.random_edge:
            # Randomly select an edge to exchange, until the subchain_length is reached.
            site_pairs = random.choices(self.graph.edges(), k=self.subchain_length)
        else:
            # We always loop over all edges.
            site_pairs = self.graph.edges() 
        for (i, j) in site_pairs: 
            if self.current_config[i] == self.current_config[j]:
                continue
            proposed_config = self.current_config.clone()
            # swap the configuration on site i and j
            temp = proposed_config[i].item()
            proposed_config[i] = proposed_config[j]
            proposed_config[j] = temp
            proposed_amp = vstate.amplitude(proposed_config).cpu()
            proposed_prob = abs(proposed_amp)**2

            try:
                acceptance_ratio = proposed_prob/current_prob
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0

            if random.random() < acceptance_ratio:
                self.current_config = proposed_config
                current_amp = proposed_amp
                current_prob = proposed_prob

        return self.current_config, current_amp
    
class MetropolisExchangeSamplerSpinful(MetropolisExchangeSampler):
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100, reset_chain=False, random_edge=False, subchain_length=None, equal_partition=True, dtype=torch.float32):
        """
        Parameters
        ----------
        hi : Hilbert
            The Hilbert space.
        graph : Graph
            Lattice graph.
        N_samples : int
            The number of samples.
        burn_in_steps : int
            The number of burn-in steps.
        reset_chain : bool
            Whether to reset the chain after each VMC step.
        random_edge : bool
            Whether to randomly select the edges in the exchange move.
        subchain_length : int
            The number of samples we discard before collecting two samples.
        equal_partition : bool
            Whether the number of samples is equal for all MPI processes. If False, we use eager sampling and must have SIZE > 1.
        dtype : torch.dtype

        """
        super().__init__(hi, graph, N_samples, burn_in_steps, reset_chain, random_edge, subchain_length, equal_partition, dtype)

    def _sample_next(self, vstate):
        """Sample the next configuration. Change the current configuration in place."""
        current_amp = vstate.amplitude(self.current_config).cpu()
        current_prob = abs(current_amp)**2
        proposed_config = self.current_config.clone()
        attempts = 0
        accepts = 0
        ind_n_map = {0:0, 1:1, 2:1, 3:2}

        if self.random_edge:
            site_pairs = random.choices(self.graph.edges(), k=self.subchain_length)
        else:
            site_pairs = self.graph.edges()

        for (i, j) in site_pairs:
            if self.current_config[i] == self.current_config[j]:
                continue
            attempts += 1
            proposed_config = self.current_config.clone()
            config_i = self.current_config[i].item()
            config_j = self.current_config[j].item()
            n_i = ind_n_map[self.current_config[i].item()]
            n_j = ind_n_map[self.current_config[j].item()]
            delta_n = abs(n_i - n_j)
            # delta_n = 1 --> SWAP
            if delta_n == 1:
                proposed_config[i] = config_j
                proposed_config[j] = config_i
                proposed_amp = vstate.amplitude(proposed_config).cpu()
                proposed_prob = abs(proposed_amp)**2
            elif delta_n == 0:
                choices = [(0, 3), (3, 0), (config_j, config_i)]
                choice = random.choice(choices)
                proposed_config[i] = choice[0]
                proposed_config[j] = choice[1]
                proposed_amp = vstate.amplitude(proposed_config).cpu()
                proposed_prob = abs(proposed_amp)**2
            elif delta_n == 2:
                choices = [(config_j, config_i), (1,2), (2,1)]
                choice = random.choice(choices)
                proposed_config[i] = choice[0]
                proposed_config[j] = choice[1]
                proposed_amp = vstate.amplitude(proposed_config).cpu()
                proposed_prob = abs(proposed_amp)**2
            else:
                raise ValueError("Invalid configuration")
            try:
                acceptance_ratio = proposed_prob/current_prob
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0

            if random.random() < acceptance_ratio or (current_prob == 0):
                self.current_config = proposed_config
                current_amp = proposed_amp
                current_prob = proposed_prob
                accepts += 1
            
        if current_amp == 0 and DEBUG:
            print(f'Rank{RANK}: Warning: psi_sigma is zero for configuration {self.current_config}, proposed_config {proposed_config}, proposed_prob {proposed_prob}')
        
        return self.current_config, current_amp