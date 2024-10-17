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

from .global_var import DEBUG, set_debug, TIME_PROFILING

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
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100, reset_chain=False, dtype=torch.float32):
        self.hi = hi
        self.Ns = N_samples
        self.graph = graph
        self.burn_in_steps = burn_in_steps
        self.reset_chain = reset_chain
        self.dtype = dtype
        self.initial_config = None
        self.current_config = None
        self.reset()
    
    def reset(self):
        """Reset the current sampler configuration to a random config in the Hilbert space."""
        rand_int = random.randint(0, 2**32-1)
        self.initial_config = torch.tensor(np.asarray(self.hi.random_state(jax.random.PRNGKey(rand_int))), dtype=self.dtype)
        self.current_config = self.initial_config.clone()

    def _sample_next(self, vstate_func):
        """Sample the next configuration."""
        raise NotImplementedError
    
    def sample(self, op, vstate, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration."""
        raise NotImplementedError
    

class MetropolisExchangeSampler(Sampler):
    """Metropolis-Hastings sampler that uses the exchange move as the sampling rule."""
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100, reset_chain=False, random_edge=False, subchain_length=None, dtype=torch.float32):
        super().__init__(hi, graph, N_samples, burn_in_steps, reset_chain)
        self.random_edge = random_edge
        self.subchain_length = hi.size if subchain_length is None else subchain_length
    
    def burn_in(self, vstate):
        """Discard the initial samples. (Burn-in)"""
        for _ in range(self.burn_in_steps):
            self._sample_next(vstate)
    
    def _sample_next(self, vstate):
        """Sample the next configuration. Change the current configuration in place.
        Must be implemented in the derived class."""
        raise NotImplementedError
    
    def sample(self, op, vstate, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration."""
        if RANK == 0:
            print('Burn-in...')
            t_burnin0 = time.time()
        with pyinstrument.Profiler() as prof:
            self.burn_in(vstate)
        if RANK == 0:
            t_burnin1 = time.time()
            print('Burn-in time:', t_burnin1 - t_burnin0)
            if TIME_PROFILING:
                prof.print(short_mode=True)
            

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
            pbar = tqdm(total=chain_length, desc='Sampling starts on rank 0...')
        
        with pyinstrument.Profiler() as prof1:
            for chain_step in range(chain_length):
                if RANK == 0:
                    time0 = time.time()

                # sample the next configuration
                sigma = self._sample_next(vstate)

                n += 1
                if RANK == 0:
                    time1 = time.time()
                    pbar.set_postfix({'Time per sample': (time1 - time0)})

                # compute local energy and amplitude gradient
                psi_sigma, logpsi_sigma_grad = vstate.amplitude_grad(sigma)

                # compute the connected non-zero operator matrix elements <eta|O|sigma>
                eta, O_etasigma = op.get_conn(sigma) # Non-zero matrix elements and corresponding configurations
                psi_eta = vstate.amplitude(eta)

                # convert torch tensors to numpy arrays
                psi_sigma = psi_sigma.detach().numpy()
                psi_eta = psi_eta.detach().numpy()
                logpsi_sigma_grad = logpsi_sigma_grad.detach().numpy()

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

                # add a progress bar if rank == 0
                if RANK == 0:
                    pbar.update(1)
        
        if RANK == 0 and TIME_PROFILING:
            print(f'MCMC sampling profiling (Chain length: {chain_length}):')
            prof1.print(short_mode=True)
        
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



class MetropolisExchangeSamplerSpinless(MetropolisExchangeSampler):
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100, reset_chain=False, random_edge=False, subchain_length=None, dtype=torch.float32):
        super().__init__(hi, graph, N_samples, burn_in_steps, reset_chain, random_edge, subchain_length, dtype)
    
    def _sample_next(self, vstate):
        """Sample the next configuration. Change the current configuration in place."""
        current_prob = abs(vstate.amplitude(self.current_config))**2
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
            proposed_prob = abs(vstate.amplitude(proposed_config))**2

            try:
                acceptance_ratio = proposed_prob/current_prob
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0

            if random.random() < acceptance_ratio:
                self.current_config = proposed_config
                current_prob = proposed_prob

        return self.current_config
    
class MetropolisExchangeSamplerSpinful(MetropolisExchangeSampler):
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100, reset_chain=False, random_edge=False, subchain_length=None, dtype=torch.float32):
        super().__init__(hi, graph, N_samples, burn_in_steps, reset_chain, random_edge, subchain_length, dtype)

    def _sample_next(self, vstate):
        """Sample the next configuration. Change the current configuration in place."""
        current_prob = abs(vstate.amplitude(self.current_config))**2
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
                proposed_prob = abs(vstate.amplitude(proposed_config))**2
            elif delta_n == 0:
                choices = [(0, 3), (3, 0), (config_j, config_i)]
                choice = random.choice(choices)
                proposed_config[i] = choice[0]
                proposed_config[j] = choice[1]
                proposed_prob = abs(vstate.amplitude(proposed_config))**2
            elif delta_n == 2:
                choices = [(config_j, config_i), (1,2), (2,1)]
                choice = random.choice(choices)
                proposed_config[i] = choice[0]
                proposed_config[j] = choice[1]
                proposed_prob = abs(vstate.amplitude(proposed_config))**2
            else:
                raise ValueError("Invalid configuration")
            try:
                acceptance_ratio = proposed_prob/current_prob
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0

            if random.random() < acceptance_ratio:
                self.current_config = proposed_config
                current_prob = proposed_prob
                accepts += 1
        
        return self.current_config