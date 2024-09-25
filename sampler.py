import numpy as np
from mpi4py import MPI
import random
from tqdm import tqdm

# torch
import torch

# quimb
from autoray import do

from global_var import DEBUG, set_debug

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class Sampler:
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100):
        self.hi = hi
        self.Ns = N_samples
        self.graph = graph
        self.burn_in_steps = burn_in_steps
        self.initial_config = torch.tensor(np.asarray(random.choice(hi.all_states())), dtype=torch.float32)
        self.current_config = self.initial_config.clone()
    
    def reset(self):
        self.initial_config = torch.tensor(np.asarray(random.choice(self.hi.all_states())), dtype=torch.float32)
        self.current_config = self.initial_config.clone()

    def _sample_next(self, vstate_func):
        """Sample the next configuration."""
        raise NotImplementedError
    
    def sample(self, op, vstate, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration."""
        raise NotImplementedError
    
class MetropolisExchangeSampler(Sampler):
    def __init__(self, hi, graph, N_samples=2**8, burn_in_steps=100):
        super().__init__(hi, graph, N_samples, burn_in_steps)
    
    def burn_in(self, vstate):
        """Discard the initial samples. (Burn-in)"""
        for _ in range(self.burn_in_steps):
            self._sample_next(vstate)
    
    def _sample_next(self, vstate):
        """Sample the next configuration. Change the current configuration in place."""
        current_prob = abs(vstate.amplitude(self.current_config))**2
        proposed_config = self.current_config.clone()
        attempts = 0
        accepts = 0
        for (i, j) in self.graph.edges(): # We always loop over all edges.
            if self.current_config[i] == self.current_config[j]:
                continue
            attempts += 1
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
                accepts += 1
        # print('Acceptance rate: {}'.format(accepts/attempts))
            
        return self.current_config
    
    def sample(self, op, vstate, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration."""
        self.burn_in(vstate)
        samples = []
        if RANK == 0:
            pbar = tqdm(total=chain_length, desc='Sampling starts on rank 0...')
        for _ in range(chain_length):
            sigma = self._sample_next(vstate)
            # compute local energy and amplitude gradient
            eta, O_etasigma = op.get_conn(sigma) # Non-zero matrix elements and corresponding configurations
            psi_sigma, log_psi_sigma_grad = vstate.amplitude_grad(sigma)
            psi_eta = vstate.amplitude(eta)

            psi_sigma = psi_sigma.detach().numpy()
            psi_eta = psi_eta.detach().numpy()
            log_psi_sigma_grad = log_psi_sigma_grad.detach().numpy()

            O_loc = np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)
            samples.append((sigma.detach().numpy(), O_loc, psi_sigma, log_psi_sigma_grad))
            # add a progress bar if rank == 0
            if RANK == 0:
                pbar.update(1)
        if RANK == 0:
            pbar.close()


        return samples