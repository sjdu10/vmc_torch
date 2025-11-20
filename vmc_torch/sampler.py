import random

import numpy as np

# torch
import torch

# quimb
from mpi4py import MPI
from tqdm import tqdm

from .global_var import DEBUG, TAG_OFFSET

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


# --- Utils ---
def from_netket_config_to_quimb_config(netket_configs):
    def func(netket_config):
        """Translate netket spin-1/2 config to tensor network product state config"""
        total_sites = len(netket_config) // 2
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


class AbstractSampler:
    def __init__(
        self,
        hi,
        graph,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        equal_partition=True,
        dtype=torch.float32,
        device=None,
        debug=False,
    ):
        self.hi = hi
        self.Ns = N_samples
        self.graph = graph
        self.burn_in_already = False
        self.burn_in_steps = burn_in_steps
        self.reset_chain = reset_chain
        self.dtype = dtype
        self.device = device
        self.initial_config = None
        self.current_config = None
        self.equal_partition = equal_partition
        self.sample_time = 0
        self.local_energy_time = 0
        self.grad_time = 0
        self.burn_in_time = 0
        self.debug = debug

        self.initial_config = torch.tensor(
            np.asarray(self.hi.random_state()), dtype=self.dtype, device=self.device
        )
        self.current_config = self.initial_config.clone()

    def reset(self):
        """Reset the current sampler configuration to a random config in the Hilbert space."""
        self.initial_config = torch.tensor(
            np.asarray(self.hi.random_state()), dtype=self.dtype, device=self.device
        )
        self.current_config = self.initial_config.clone()

    def _sample_next(self, vstate_func):
        """Sample the next configuration."""
        raise NotImplementedError

    def sample(self, vstate, op, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration."""
        raise NotImplementedError


class Sampler(AbstractSampler):
    """Markov Chain sampler"""

    def __init__(
        self,
        hi,
        graph,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        random_edge=False,
        subchain_length=None,
        equal_partition=True,
        dtype=torch.float32,
        device=None,
        debug=False,
    ):
        super().__init__(
            hi,
            graph,
            N_samples,
            burn_in_steps,
            reset_chain,
            equal_partition,
            dtype,
            device,
            debug,
        )
        self.random_edge = random_edge
        self.subchain_length = (
            graph.n_edges if subchain_length is None else subchain_length
        )
        self.attempts = 0
        self.accepts = 0
        self.hopping_rate = 0.0

    @torch.no_grad()
    def burn_in(self, vstate):
        """Discard the initial samples. (Burn-in)"""
        if self.burn_in_already and not self.reset_chain:
            """Avoid multiple burn-in calls."""
            return
        for _ in range(self.burn_in_steps):
            self._sample_next(vstate, burn_in=True)
        self.burn_in_already = True

    @torch.no_grad()
    def _sample_next(self, vstate, **kwargs):
        """Sample the next configuration. Change the current configuration in place.
        Must be implemented in the derived class."""
        raise NotImplementedError

    @torch.no_grad()
    def sample_configs(self, vstate, chain_length=1, iprint=0):
        """
        Sample configurations.
        Returns
        -------
        configs : list
            The sampled configurations.
        amps : list
            Amplitudes of the sampled configurations.
        """
        if iprint:
            print("Burn-in...")
            t_burnin0 = MPI.Wtime()
        self.burn_in(vstate)
        if iprint:
            t_burnin1 = MPI.Wtime()
            print("Burn-in time:", t_burnin1 - t_burnin0)

        configs = torch.zeros(
            (chain_length, self.graph.N), dtype=self.dtype, device=self.device
        )
        amps = torch.zeros((chain_length,), dtype=self.dtype, device=self.device)
        for _ in range(chain_length):
            # sample the next configuration
            psi_sigma = 0
            while psi_sigma == 0:
                # We need to make sure that the amplitude is not zero
                sigma, psi_sigma = self._sample_next(vstate, burn_in=True)
            configs[_] = sigma
            amps[_] = psi_sigma

        return configs, amps

    @torch.no_grad()
    def sample_configs_eager(self, vstate, message_tag=None, **kwargs):
        """Sample eagerly for the configs and amplitudes.
        return two tensors: configs and amps"""
        assert not self.equal_partition, (
            "Must not use equal partition for eager sampling."
        )

        if RANK == 0:
            print("Burn-in...")
        t_burnin0 = MPI.Wtime()
        self.burn_in(vstate)
        t_burnin1 = MPI.Wtime()
        if RANK == 0:
            print("Burn-in time:", t_burnin1 - t_burnin0)
        self.burn_in_time = t_burnin1 - t_burnin0

        # We only estimate the variance of op_loc locally
        n = 0
        n_total = 0
        sigma_vec = []
        amp_vec = []
        terminate = np.array([0], dtype=np.int32)

        if RANK == 0:
            pbar = tqdm(total=self.Ns, desc="Sampling starts...")
            for _ in range(2):
                sigma, psi_sigma = self._sample_next(vstate, burn_in=True)
                sigma_vec.append(sigma.cpu().numpy())
                amp_vec.append(psi_sigma.cpu().numpy())
                n += 1
                n_total += 1
                pbar.update(1)

            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET - 1):
                redundant_message = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [redundant_message, MPI.INT],
                    source=MPI.ANY_SOURCE,
                    tag=message_tag + TAG_OFFSET - 1,
                )
                del redundant_message

            while not terminate[0]:
                # Receive the local sample count from each rank
                buf = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [buf, MPI.INT], source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET
                )
                dest_rank = buf[0]
                n_total += 1
                pbar.update(1)
                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = np.array([1], dtype=np.int32)
                    for dest_rank in range(1, SIZE):
                        COMM.Send(
                            [terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1
                        )
                # Send the termination signal to the rank
                COMM.Send([terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1)

            pbar.close()

        else:
            while not terminate[0]:
                sigma, psi_sigma = self._sample_next(vstate, burn_in=True)
                sigma_vec.append(sigma.cpu().numpy())
                amp_vec.append(psi_sigma.cpu().numpy())
                n += 1

                buf = np.array([RANK], dtype=np.int32)
                # Send the local sample count to rank 0
                # COMM.send(buf, dest=0, tag=message_tag+TAG_OFFSET)
                COMM.Send([buf, MPI.INT], dest=0, tag=message_tag + TAG_OFFSET)
                # Receive the termination signal from rank 0
                terminate = np.empty(1, dtype=np.int32)
                COMM.Recv([terminate, MPI.INT], source=0, tag=message_tag + 1)

        vstate.clear_env_cache()

        configs = torch.tensor(
            np.array(sigma_vec), dtype=self.dtype, device=self.device
        )
        amps = torch.tensor(np.array(amp_vec), dtype=self.dtype, device=self.device)

        return configs, amps

    def sample(self, vstate, op, chain_length=1, vec_op=False, pgbar=True, **kwargs):
        """
        Sample configurations and compute the local operator.

        Returns
        -------
        op_loc_sum : float
            The sum of the local operator.
        op_loc_var : float
            The variance of the local operator.
        W_loc : float
            The within chain variance of the local operator.
        chain_means_loc : list
            The means of the local operator for each chain.

        """
        assert self.equal_partition, (
            "The number of samples must be equal for all MPI processes."
        )
        if RANK == 0:
            print("Burn-in...")
        t_burnin0 = MPI.Wtime()
        self.burn_in(vstate)
        t_burnin1 = MPI.Wtime()
        if RANK == 0:
            print("Burn-in time:", t_burnin1 - t_burnin0)
        self.burn_in_time = t_burnin1 - t_burnin0

        op_loc_sum = 0

        # We use Welford's Algorithm to compute the sample variance of op_loc in a single pass.
        n = 0
        op_loc_mean = 0
        op_loc_M2 = 0
        op_loc_var = 0
        op_loc_vec = np.zeros(chain_length)

        if RANK == 0 and pgbar:
            pbar_dummy = tqdm(total=0, bar_format="MCMC sampling for <O>:", leave=False)
            pbar = tqdm(total=chain_length, desc="Sampling starts for rank 0...")

        for chain_step in range(chain_length):
            time0 = MPI.Wtime()
            # sample the next configuration
            psi_sigma = 0
            with torch.no_grad():
                while psi_sigma == 0:
                    # We need to make sure that the amplitude is not zero
                    sigma, psi_sigma = self._sample_next(vstate)

            n += 1

            # compute local energy and amplitude gradient
            vstate.set_cache_env_mode(on=True)
            psi_sigma = vstate.amplitude(sigma, _cache=1)
            vstate.set_cache_env_mode(on=False)
            # compute the connected non-zero operator matrix elements <eta|O|sigma>
            eta, O_etasigma = op.get_conn(
                sigma
            )  # Non-zero matrix elements and corresponding configurations
            psi_eta = vstate.amplitude(eta)

            # convert torch tensors to numpy arrays
            psi_sigma = psi_sigma.cpu().detach().numpy()
            psi_eta = psi_eta.cpu().detach().numpy()

            # compute the local operator
            op_loc = (
                np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)
                if not vec_op
                else O_etasigma * (psi_eta / psi_sigma)
            )
            if not vec_op:
                op_loc_vec[chain_step] = op_loc

            # accumulate the local operator
            op_loc_sum += op_loc

            # update the sample variance of op_loc
            op_loc_mean_prev = op_loc_mean
            op_loc_mean += (op_loc - op_loc_mean) / n
            op_loc_M2 += (op_loc - op_loc_mean_prev) * (op_loc - op_loc_mean)

            time1 = MPI.Wtime()
            if RANK == 0 and pgbar:
                pbar.set_postfix({"Time per sample for each rank": (time1 - time0)})

            # update the sample variance
            if n > 1:
                op_loc_var = op_loc_M2 / (n - 1)

            # add a progress bar if rank == 0
            if RANK == 0 and pgbar:
                pbar.update(1)

        vstate.clear_env_cache()

        if self.reset_chain:
            self.reset()

        if RANK == 0 and pgbar:
            pbar.close()
            pbar_dummy.close()

        if not vec_op:
            # The following is for computing the Rhat diagnostic using the Gelman-Rubin formula
            # Step 1：split the chain in half and compute the within chain variance
            split_chains = np.split(op_loc_vec, 2)
            # Step 2: compute the within chain variance
            W_loc = np.sum([np.var(split_chain) for split_chain in split_chains])
            chain_means_loc = [np.mean(split_chain) for split_chain in split_chains]
        else:
            W_loc = None
            chain_means_loc = None

        samples = (op_loc_sum, op_loc_var, W_loc, chain_means_loc)
        self.attempts = 0
        self.accepts = 0
        return samples

    @torch.no_grad()
    def sample_eager(self, vstate, op, message_tag=None, pgbar=False, **kwargs):
        """Sample eagerly for the local energy and amplitude gradient for each configuration.
        return a tuple of (op_loc_sum, op_loc_var, n)"""
        assert not self.equal_partition, (
            "Must not use equal partition for eager sampling."
        )

        if RANK == 0:
            print("Burn-in...")
        t_burnin0 = MPI.Wtime()
        self.burn_in(vstate)
        t_burnin1 = MPI.Wtime()
        if RANK == 0:
            print("Burn-in time:", t_burnin1 - t_burnin0)
        self.burn_in_time = t_burnin1 - t_burnin0

        op_loc_sum = 0

        # We only estimate the variance of op_loc locally
        n = 0
        n_total = 0
        op_loc_vec = []
        terminate = np.array([0], dtype=np.int32)

        if RANK == 0:
            if pgbar:
                pbar_dummy = tqdm(
                    total=0, bar_format="MCMC sampling for <O>:", leave=False
                )
                pbar = tqdm(total=self.Ns, desc="Sampling starts...")
            else:
                pbar = None
            for _ in range(2):
                op_loc, _ = self._sample_expect(vstate, op, **kwargs)
                op_loc_vec.append(op_loc)
                op_loc_sum += op_loc

                n += 1
                n_total += 1
                if pgbar:
                    pbar.update(1)
                    pbar.set_postfix("")
                    pbar.refresh()

            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET - 1):
                redundant_message = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [redundant_message, MPI.INT],
                    source=MPI.ANY_SOURCE,
                    tag=message_tag + TAG_OFFSET - 1,
                )
                del redundant_message

            while not terminate[0]:
                # Receive the local sample count from each rank
                buf = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [buf, MPI.INT], source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET
                )
                dest_rank = buf[0]
                n_total += 1
                if pgbar:
                    pbar.update(1)
                    pbar.set_postfix("")
                    pbar.refresh()

                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = np.array([1], dtype=np.int32)
                    for dest_rank in range(1, SIZE):
                        COMM.Send(
                            [terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1
                        )
                # Send the termination signal to the rank
                COMM.Send([terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1)

            if pgbar:
                pbar.close()
                pbar_dummy.close()

        else:
            while not terminate[0]:
                op_loc, _ = self._sample_expect(vstate, op, **kwargs)
                n += 1
                op_loc_vec.append(op_loc)
                # accumulate the local energy and amplitude gradient
                op_loc_sum += op_loc

                buf = np.array([RANK], dtype=np.int32)
                # Send the local sample count to rank 0
                # COMM.send(buf, dest=0, tag=message_tag+TAG_OFFSET)
                COMM.Send([buf, MPI.INT], dest=0, tag=message_tag + TAG_OFFSET)
                # Receive the termination signal from rank 0
                terminate = np.empty(1, dtype=np.int32)
                COMM.Recv([terminate, MPI.INT], source=0, tag=message_tag + 1)

        vstate.clear_env_cache()

        if self.reset_chain:
            self.reset()

        # convert the list to numpy array
        op_loc_vec = np.asarray(op_loc_vec)
        if n > 1:
            op_loc_var = np.var(op_loc_vec, ddof=1)
        else:
            op_loc_var = 0
        samples = (op_loc_sum, op_loc_var, n)
        self.attempts = 0
        self.accepts = 0
        return samples

    def sample_w_grad(self, vstate, op, chain_length=1):
        """Sample the local energy and amplitude gradient for each configuration.
        return a tuple of (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat)"""
        assert self.equal_partition, (
            "The number of samples must be equal for all MPI processes."
        )
        if RANK == 0:
            print("Burn-in...")
        t_burnin0 = MPI.Wtime()
        self.burn_in(vstate)
        t_burnin1 = MPI.Wtime()
        if RANK == 0:
            print("Burn-in time:", t_burnin1 - t_burnin0)
        self.burn_in_time = t_burnin1 - t_burnin0

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
            pbar = tqdm(total=chain_length, desc="Sampling starts for rank 0...")

        for chain_step in range(chain_length):
            time0 = MPI.Wtime()
            # sample the next configuration
            psi_sigma = 0
            with torch.no_grad():
                while psi_sigma == 0:
                    # We need to make sure that the amplitude is not zero
                    sigma, psi_sigma = self._sample_next(vstate)

            time1 = MPI.Wtime()
            n += 1
            if RANK == 0:
                pbar.set_postfix({"Time per sample for each rank": (time1 - time0)})

            # compute local energy and amplitude gradient
            vstate.set_cache_env_mode(on=True)
            psi_sigma, logpsi_sigma_grad = vstate.amplitude_grad(sigma)
            vstate.set_cache_env_mode(on=False)
            time2 = MPI.Wtime()
            # compute the connected non-zero operator matrix elements <eta|O|sigma>
            eta, O_etasigma = op.get_conn(
                sigma
            )  # Non-zero matrix elements and corresponding configurations
            psi_eta = vstate.amplitude(eta)
            time3 = MPI.Wtime()
            # if RANK==1:
            #     print('Rank1 local energy + gradient time:', time3 - time1, 'Number of contractions:', len(eta))

            self.sample_time += time1 - time0  # for profiling purposes
            self.local_energy_time += time3 - time2
            self.grad_time += time2 - time1

            # convert torch tensors to numpy arrays
            psi_sigma = psi_sigma.cpu().detach().numpy()
            psi_eta = psi_eta.cpu().detach().numpy()
            logpsi_sigma_grad = logpsi_sigma_grad.cpu().detach().numpy()

            # compute the local operator
            op_loc = np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)
            # if abs(op_loc) > 1e4:
            #     chain_step -= 1
            #     continue
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

        vstate.clear_env_cache()

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

        samples = (
            op_loc_sum,
            logpsi_sigma_grad_sum,
            op_logpsi_sigma_grad_product_sum,
            op_loc_var,
            logpsi_sigma_grad_mat,
            W_loc,
            chain_means_loc,
        )
        self.attempts = 0
        self.accepts = 0
        return samples

    def sample_w_grad_eager(self, vstate, op, message_tag=None):
        """Sample eagerly for the local energy and amplitude gradient for each configuration.
        return a tuple of (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat)"""
        assert not self.equal_partition, (
            "Must not use equal partition for eager sampling."
        )

        if RANK == 0:
            print("Burn-in...")
        t_burnin0 = MPI.Wtime()
        self.burn_in(vstate)
        t_burnin1 = MPI.Wtime()
        if RANK == 0:
            print("Burn-in time:", t_burnin1 - t_burnin0)
        self.burn_in_time = t_burnin1 - t_burnin0

        op_loc_sum = 0
        logpsi_sigma_grad_sum = np.zeros(vstate.Np)
        op_logpsi_sigma_grad_product_sum = np.zeros(vstate.Np)

        logpsi_sigma_grad_mat = []
        # We only estimate the variance of op_loc locally
        n = 0
        n_total = 0
        op_loc_vec = []
        # terminate = False
        terminate = np.array([0], dtype=np.int32)

        if RANK == 0:
            pbar = tqdm(total=self.Ns, desc="Sampling starts...")
            for _ in range(1):
                op_loc, logpsi_sigma_grad, _ = self._sample_expect_grad(vstate, op)
                op_loc_vec.append(op_loc)
                op_loc_sum += op_loc
                logpsi_sigma_grad_sum += logpsi_sigma_grad  # summed to save memory ?
                # XXX: summing is not necessary, we can put all post-processing of the samples in variational_state module
                op_logpsi_sigma_grad_product_sum += op_loc * logpsi_sigma_grad
                logpsi_sigma_grad_mat.append(logpsi_sigma_grad)

                n += 1
                n_total += 1
                pbar.update(1)

            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET - 1):
                redundant_message = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [redundant_message, MPI.INT],
                    source=MPI.ANY_SOURCE,
                    tag=message_tag + TAG_OFFSET - 1,
                )
                # redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1)
                del redundant_message

            while not terminate[0]:
                # Receive the local sample count from each rank
                buf = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [buf, MPI.INT], source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET
                )
                dest_rank = buf[0]
                # buf = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET)
                # dest_rank = buf[0]
                n_total += 1
                pbar.update(1)
                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = np.array([1], dtype=np.int32)
                    for dest_rank in range(1, SIZE):
                        # COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                        COMM.Send(
                            [terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1
                        )
                # Send the termination signal to the rank
                # COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                COMM.Send([terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1)

            pbar.close()

        else:
            while not terminate[0]:
                try:
                    op_loc, logpsi_sigma_grad, _ = self._sample_expect_grad(vstate, op)
                except Exception as e:
                    print(
                        f"    RANK{RANK} Error in sampling: {e}, setting local energy and gradient to zero."
                    )
                    op_loc, logpsi_sigma_grad = 0, np.zeros(vstate.Np)

                if abs(op_loc) > 1e8:  # discard the extreme samples
                    continue
                n += 1
                op_loc_vec.append(op_loc)
                # accumulate the local energy and amplitude gradient
                op_loc_sum += op_loc
                logpsi_sigma_grad_sum += logpsi_sigma_grad
                op_logpsi_sigma_grad_product_sum += op_loc * logpsi_sigma_grad

                # collect the log-amplitude gradient
                logpsi_sigma_grad_mat.append(logpsi_sigma_grad)

                # buf = (RANK,)
                buf = np.array([RANK], dtype=np.int32)
                # Send the local sample count to rank 0
                # COMM.send(buf, dest=0, tag=message_tag+TAG_OFFSET)
                COMM.Send([buf, MPI.INT], dest=0, tag=message_tag + TAG_OFFSET)
                # Receive the termination signal from rank 0
                # terminate = COMM.recv(source=0, tag=message_tag+1)
                terminate = np.empty(1, dtype=np.int32)
                COMM.Recv([terminate, MPI.INT], source=0, tag=message_tag + 1)

        vstate.clear_env_cache()

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
        samples = (
            op_loc_sum,
            logpsi_sigma_grad_sum,
            op_logpsi_sigma_grad_product_sum,
            op_loc_var,
            logpsi_sigma_grad_mat,
            op_loc_vec,
        )
        self.attempts = 0
        self.accepts = 0
        return samples

    def _sample_expect_grad(self, vstate, op):
        """Get one sample of the local energy and amplitude gradient."""
        time0 = MPI.Wtime()
        # sample the next configuration
        psi_sigma = 0
        with torch.no_grad():
            iter = 0
            while psi_sigma == 0:
                # We need to make sure that the amplitude is not zero
                sigma, psi_sigma = self._sample_next(vstate)
                iter += 1
                if iter > 50:
                    print(
                        f"    RANK{RANK} Warning: Exceeded 50 attempts to find non-zero amplitude. Break out."
                    )
                    break
        time1 = MPI.Wtime()

        # compute local energy and amplitude gradient
        vstate.set_cache_env_mode(on=True)
        _, logpsi_sigma_grad = vstate.amplitude_grad(sigma)
        vstate.set_cache_env_mode(on=False)

        # # if DEBUG:
        if logpsi_sigma_grad.abs().max() > 1e5:
            logpsi_sigma_grad = torch.clamp(logpsi_sigma_grad, min=-1e5, max=1e5)
            # # do clipping to the large values in the gradient: set those values to 0
            # logpsi_sigma_grad = torch.where(logpsi_sigma_grad.abs() > 1e5, torch.zeros_like(logpsi_sigma_grad), logpsi_sigma_grad)

        # compute the connected non-zero operator matrix elements <eta|O|sigma>
        time2 = MPI.Wtime()
        eta, O_etasigma = op.get_conn(sigma)
        psi_eta = vstate.amplitude(eta)
        vstate.clear_env_cache()
        time3 = MPI.Wtime()
        # if RANK==1:
        #     print(f"    RANK1 sample {sigma}, Local Energy size: {len(eta)}")
        # convert torch tensors to numpy arrays
        psi_sigma = psi_sigma.cpu().detach().numpy()
        psi_eta = psi_eta.cpu().detach().numpy()
        logpsi_sigma_grad = logpsi_sigma_grad.cpu().detach().numpy()

        # compute the local operator
        op_loc = np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)

        if DEBUG:
            if abs(op_loc) > 1e4:
                print(
                    f"    RANK{RANK} Local Energy: {op_loc} Amplitude: {psi_sigma.item():.10g} Gradient mean: {np.mean(abs(logpsi_sigma_grad))}"
                )

        self.sample_time += time1 - time0  # for profiling purposes
        self.local_energy_time += time3 - time2
        self.grad_time += time2 - time1

        return op_loc, logpsi_sigma_grad, time1 - time0

    @torch.no_grad()
    def _sample_expect(self, vstate, op, vec_op=False):
        """Get one sample of the local operator."""
        time0 = MPI.Wtime()
        # sample the next configuration
        psi_sigma = 0
        with torch.no_grad():
            while psi_sigma == 0:
                # We need to make sure that the amplitude is not zero
                sigma, psi_sigma = self._sample_next(vstate)
        time1 = MPI.Wtime()

        # compute local energy and amplitude gradient
        vstate.set_cache_env_mode(on=True)
        psi_sigma = vstate.amplitude(sigma, _cache=1)
        vstate.set_cache_env_mode(on=False)
        time2 = MPI.Wtime()
        # compute the connected non-zero operator matrix elements <eta|O|sigma>
        eta, O_etasigma = op.get_conn(sigma)
        psi_eta = vstate.amplitude(eta)
        vstate.clear_env_cache()
        time3 = MPI.Wtime()

        # convert torch tensors to numpy arrays
        psi_sigma = psi_sigma.cpu().detach().numpy()
        psi_eta = psi_eta.cpu().detach().numpy()

        # compute the local operator
        op_loc = (
            np.sum(O_etasigma * (psi_eta / psi_sigma), axis=-1)
            if not vec_op
            else O_etasigma * (psi_eta / psi_sigma)
        )

        self.sample_time += time1 - time0  # for profiling purposes
        self.local_energy_time += time3 - time2
        self.grad_time += time2 - time1

        return op_loc, time1 - time0

    def sample_expectation(self, vstate, op, chain_length=1):
        """Sample the expectation value of the operator `op`, using single-chain sampling."""

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
        expect_op = (
            psi_vec.conj().T @ (op_dense @ psi_vec) / (psi_vec.conj().T @ psi_vec)
        )
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
            print("Burn-in...")
        t_burnin0 = MPI.Wtime()
        self.burn_in(vstate)
        t_burnin1 = MPI.Wtime()
        if RANK == 0:
            print("Burn-in time:", t_burnin1 - t_burnin0)
        self.burn_in_time = t_burnin1 - t_burnin0

        n = 0
        n_total = 0
        config_list = []
        config_amplitudes_dict = {}
        terminate = np.array([0], dtype=np.int32)

        if RANK == 0:
            pbar = tqdm(total=self.Ns, desc="Sampling starts...")
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
                    psi_sigma, op_psi_eta = config_amplitudes_dict[
                        config
                    ]  # XXX: not needed actually
                n += 1
                n_total += 1
                pbar.update(1)

            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET - 1):
                redundant_message = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [redundant_message, MPI.INT],
                    source=MPI.ANY_SOURCE,
                    tag=message_tag + TAG_OFFSET - 1,
                )
                # redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1)
                del redundant_message

            while not terminate[0]:
                # Receive the local sample count from each rank
                buf = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [buf, MPI.INT], source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET
                )
                # buf = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET)
                # dest_rank = buf[0]
                dest_rank = buf[0]
                n_total += 1
                pbar.update(1)
                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = np.array([1], dtype=np.int32)
                    for dest_rank in range(1, SIZE):
                        # COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                        COMM.Send(
                            [terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1
                        )
                # Send the termination signal to the rank
                # COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                COMM.Send([terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1)

            pbar.close()

        else:
            while not terminate[0]:
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

                # buf = (RANK,)
                buf = np.array([RANK], dtype=np.int32)
                # Send the local sample count to rank 0
                COMM.Send([buf, MPI.INT], dest=0, tag=message_tag + TAG_OFFSET)
                # Receive the termination signal from rank 0
                # terminate = COMM.recv(source=0, tag=message_tag+1)
                terminate = np.empty(1, dtype=np.int32)
                COMM.Recv([terminate, MPI.INT], source=0, tag=message_tag + 1)

        if self.reset_chain:
            self.reset()

        if RANK == 0:
            pbar.close()

        dataset = (config_list, config_amplitudes_dict)

        return dataset

    def sample_SWO_state_fitting_dataset_eager(
        self, vstate, target_state, message_tag=None, compute_energy=False, op=None
    ):
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
            print("Burn-in...")
        t_burnin0 = MPI.Wtime()
        self.burn_in(vstate)
        t_burnin1 = MPI.Wtime()
        if RANK == 0:
            print("Burn-in time:", t_burnin1 - t_burnin0)
        self.burn_in_time = t_burnin1 - t_burnin0

        n = 0
        n_total = 0
        config_list = []
        config_amplitudes_dict = {}
        terminate = np.array([0], dtype=np.int32)
        op_expect = 0

        if RANK == 0:
            pbar = tqdm(total=self.Ns, desc="Sampling starts...")
            for _ in range(2):
                config, psi_sigma = self._sample_next(vstate)
                config_list.append(config)
                if config not in config_amplitudes_dict:
                    psi_target_sigma = target_state.amplitude(config)
                    config_amplitudes_dict[config] = (psi_sigma, psi_target_sigma)
                else:
                    psi_sigma, psi_target_sigma = config_amplitudes_dict[
                        config
                    ]  # XXX: not needed actually
                if compute_energy:
                    eta, O_etasigma = op.get_conn(config)
                    psi_eta = vstate.amplitude(eta)
                    psi_eta = psi_eta.cpu().detach().numpy()
                    psi_sigma = psi_sigma.cpu().detach().numpy()
                    op_loc = np.sum(O_etasigma * psi_eta / psi_sigma, axis=-1)
                    op_expect += op_loc
                    ...  # XXX: compute variance

                n += 1
                n_total += 1
                pbar.update(1)

            # Discard messages from previous steps
            while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET - 1):
                redundant_message = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [redundant_message, MPI.INT],
                    source=MPI.ANY_SOURCE,
                    tag=message_tag + TAG_OFFSET - 1,
                )
                # redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1)
                del redundant_message

            while not terminate[0]:
                # Receive the local sample count from each rank
                # buf = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET)
                # dest_rank = buf[0]
                buf = np.empty(1, dtype=np.int32)
                COMM.Recv(
                    [buf, MPI.INT], source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET
                )
                dest_rank = buf[0]
                n_total += 1
                pbar.update(1)
                # Check if we have enough samples
                if n_total >= self.Ns:
                    terminate = np.array([1], dtype=np.int32)
                    for dest_rank in range(1, SIZE):
                        # COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                        COMM.Send(
                            [terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1
                        )
                # Send the termination signal to the rank
                # COMM.send(terminate, dest=dest_rank, tag=message_tag+1)
                COMM.Send([terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1)

            pbar.close()

        else:
            while not terminate[0]:
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
                    ...  # XXX: compute variance
                n += 1

                # buf = (RANK,)
                buf = np.array([RANK], dtype=np.int32)
                # Send the local sample count to rank 0
                # COMM.send(buf, dest=0, tag=message_tag+TAG_OFFSET)
                COMM.Send([buf, MPI.INT], dest=0, tag=message_tag + TAG_OFFSET)
                # Receive the termination signal from rank 0
                # terminate = COMM.recv(source=0, tag=message_tag+1)
                terminate = np.empty(1, dtype=np.int32)
                COMM.Recv([terminate, MPI.INT], source=0, tag=message_tag + 1)

        if self.reset_chain:
            self.reset()

        if RANK == 0:
            pbar.close()

        dataset = (config_list, config_amplitudes_dict, op_expect)

        return dataset


class MetropolisExchangeSamplerSpinless(Sampler):
    def __init__(
        self,
        hi,
        graph,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        random_edge=False,
        subchain_length=None,
        equal_partition=True,
        dtype=torch.float32,
    ):
        super().__init__(
            hi,
            graph,
            N_samples,
            burn_in_steps,
            reset_chain,
            random_edge,
            subchain_length,
            equal_partition,
            dtype,
        )

    def _sample_next(self, vstate, **kwargs):
        """Sample the next configuration. Change the current configuration in place."""
        current_amp = vstate.amplitude(self.current_config)
        current_prob = abs(current_amp) ** 2
        proposed_config = self.current_config.clone()
        if self.random_edge:
            # Randomly select an edge to exchange, until the subchain_length is reached.
            site_pairs = random.choices(self.graph.edges(), k=self.subchain_length)
        else:
            # We always loop over all edges.
            site_pairs = self.graph.edges()
        for i, j in site_pairs:
            if self.current_config[i] == self.current_config[j]:
                continue
            proposed_config = self.current_config.clone()
            # swap the configuration on site i and j
            temp = proposed_config[i].item()
            proposed_config[i] = proposed_config[j]
            proposed_config[j] = temp
            proposed_amp = vstate.amplitude(proposed_config).cpu()
            proposed_prob = abs(proposed_amp) ** 2

            try:
                acceptance_ratio = proposed_prob / current_prob
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0

            if random.random() < acceptance_ratio:
                self.current_config = proposed_config
                current_amp = proposed_amp
                current_prob = proposed_prob

        return self.current_config, current_amp


class MetropolisSamplerSpinless(Sampler):
    """
    Metropolis sampler with single-site updates for spinless fermions or bosons.
    This sampler does not conserve particle number.
    """

    def __init__(
        self,
        hi,
        graph,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        random_edge=False,
        random_site=False,
        subchain_length=None,
        equal_partition=True,
        dtype=torch.float32,
    ):
        super().__init__(
            hi,
            graph,
            N_samples,
            burn_in_steps,
            reset_chain,
            random_edge,
            subchain_length,
            equal_partition,
            dtype,
        )
        self.random_site = random_site

    def _sample_next(self, vstate, **kwargs):
        """Sample the next configuration. Change the current configuration in place."""
        current_amp = vstate.amplitude(self.current_config)
        current_prob = abs(current_amp) ** 2
        proposed_config = self.current_config.clone()
        if self.random_site:
            # Randomly select a site to update, until the subchain_length is reached.
            site_list = random.choices(range(self.graph.N), k=self.subchain_length)
        else:
            # We always loop over all sites.
            site_list = range(self.graph.N)
        for i in site_list:
            # Propose a new state for site i
            current_state = self.current_config[i]
            new_state = 1 - current_state  # flip between 0 and 1
            proposed_config = self.current_config.clone()
            proposed_config[i] = new_state
            proposed_amp = vstate.amplitude(proposed_config).cpu()
            proposed_prob = abs(proposed_amp) ** 2

            try:
                acceptance_ratio = proposed_prob / current_prob
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0

            if random.random() < acceptance_ratio:
                self.current_config = proposed_config
                current_amp = proposed_amp
                current_prob = proposed_prob

        return self.current_config, current_amp


class MetropolisExchangeSamplerSpinful(Sampler):
    def __init__(
        self,
        hi,
        graph,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        random_edge=False,
        subchain_length=None,
        equal_partition=True,
        dtype=torch.float32,
        device=None,
    ):
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
        super().__init__(
            hi,
            graph,
            N_samples,
            burn_in_steps,
            reset_chain,
            random_edge,
            subchain_length,
            equal_partition,
            dtype,
            device=device,
        )

    def _sample_next(self, vstate, **kwargs):
        """Sample the next configuration. Change the current configuration in place."""
        self.current_amp = vstate.amplitude(self.current_config)
        self.current_prob = abs(self.current_amp) ** 2
        proposed_config = self.current_config.clone()
        ind_n_map = {0: 0, 1: 1, 2: 1, 3: 2}

        def exchange_propose(i, j):
            if self.current_config[i] == self.current_config[j]:
                return
            self.attempts += 1
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
                proposed_prob = abs(proposed_amp) ** 2
            elif delta_n == 0:
                choices = [(0, 3), (3, 0), (config_j, config_i)]
                choice = random.choice(choices)
                proposed_config[i] = choice[0]
                proposed_config[j] = choice[1]
                proposed_amp = vstate.amplitude(proposed_config).cpu()
                proposed_prob = abs(proposed_amp) ** 2
            elif delta_n == 2:
                choices = [(config_j, config_i), (1, 2), (2, 1)]
                choice = random.choice(choices)
                proposed_config[i] = choice[0]
                proposed_config[j] = choice[1]
                proposed_amp = vstate.amplitude(proposed_config).cpu()
                proposed_prob = abs(proposed_amp) ** 2
            else:
                raise ValueError("Invalid configuration")
            try:
                acceptance_ratio = min(1, (proposed_prob / self.current_prob))
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0
            if random.random() < acceptance_ratio or (self.current_prob == 0):
                self.current_config = proposed_config
                self.current_amp = proposed_amp
                self.current_prob = proposed_prob
                self.accepts += 1

        if self.random_edge:
            site_pairs = random.choices(self.graph.edges(), k=self.subchain_length)
        else:
            site_pairs = self.graph.edges()

        for i, j in site_pairs:
            exchange_propose(i, j)

        if self.current_amp == 0 and DEBUG:
            print(
                f"Rank{RANK}: Warning: psi_sigma is zero for configuration {self.current_config}, proposed_config {proposed_config}, proposed_prob {abs(vstate.amplitude(proposed_config))}**2"
            )

        return self.current_config, self.current_amp


class MetropolisExchangeSamplerSpinful_hopping(Sampler):
    def __init__(
        self,
        hi,
        graph,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        random_edge=False,
        subchain_length=None,
        equal_partition=True,
        dtype=torch.float32,
        device=None,
        hopping_rate=0.5,
    ):
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
        super().__init__(
            hi,
            graph,
            N_samples,
            burn_in_steps,
            reset_chain,
            random_edge,
            subchain_length,
            equal_partition,
            dtype,
            device=device,
        )
        self.hopping_rate = hopping_rate

    def _sample_next(self, vstate, **kwargs):
        """Sample the next configuration. Change the current configuration in place."""
        self.current_amp = vstate.amplitude(self.current_config)
        self.current_prob = abs(self.current_amp) ** 2
        proposed_config = self.current_config.clone()
        ind_n_map = {0: 0, 1: 1, 2: 1, 3: 2}

        def exchange_propose(i, j):
            if self.current_config[i] == self.current_config[j]:
                self.attempts += 1
                self.accepts += 1  # exchange must be accepted, no hopping is allowed
                return
            self.attempts += 1
            proposed_config = self.current_config.clone()
            config_i = self.current_config[i].item()
            config_j = self.current_config[j].item()
            if random.random() < 1 - self.hopping_rate:
                # exchange
                proposed_config[i] = config_j
                proposed_config[j] = config_i
                proposed_amp = vstate.amplitude(proposed_config).cpu()
                proposed_prob = abs(proposed_amp) ** 2
            else:
                # hopping
                n_i = ind_n_map[self.current_config[i].item()]
                n_j = ind_n_map[self.current_config[j].item()]
                delta_n = abs(n_i - n_j)
                if delta_n == 1:
                    # consider only valid hopping: (0, u) -> (u, 0); (d, ud) -> (ud, d)
                    proposed_config[i] = config_j
                    proposed_config[j] = config_i
                    proposed_amp = vstate.amplitude(proposed_config).cpu()
                    proposed_prob = abs(proposed_amp) ** 2
                elif delta_n == 0:
                    # consider only valid hopping: (u, d) -> (0, ud) or (ud, 0)
                    choices = [(0, 3), (3, 0)]
                    choice = random.choice(choices)
                    proposed_config[i] = choice[0]
                    proposed_config[j] = choice[1]
                    proposed_amp = vstate.amplitude(proposed_config).cpu()
                    proposed_prob = abs(proposed_amp) ** 2
                elif delta_n == 2:
                    # consider only valid hopping: (0, ud) -> (u, d) or (d, u)
                    choices = [(1, 2), (2, 1)]
                    choice = random.choice(choices)
                    proposed_config[i] = choice[0]
                    proposed_config[j] = choice[1]
                    proposed_amp = vstate.amplitude(proposed_config).cpu()
                    proposed_prob = abs(proposed_amp) ** 2
                else:
                    raise ValueError("Invalid configuration")
            try:
                acceptance_ratio = min(1, (proposed_prob / self.current_prob))
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0
            if random.random() < acceptance_ratio or (self.current_prob == 0):
                self.current_config = proposed_config
                self.current_amp = proposed_amp
                self.current_prob = proposed_prob
                self.accepts += 1

        if self.random_edge:
            site_pairs = random.choices(self.graph.edges(), k=self.subchain_length)
        else:
            site_pairs = self.graph.edges()

        acceptance_rate = 0

        while acceptance_rate < 0.05:
            for i, j in site_pairs:
                exchange_propose(i, j)
            acceptance_rate = self.accepts / self.attempts if self.attempts > 0 else 0

        if self.current_amp == 0 and DEBUG:
            print(
                f"Rank{RANK}: Warning: psi_sigma is zero for configuration {self.current_config}, proposed_config {proposed_config}, proposed_prob {abs(vstate.amplitude(proposed_config))}**2"
            )

        return self.current_config, self.current_amp


class MetropolisExchangeSamplerSpinful_1D_reusable(Sampler):
    def __init__(
        self,
        hi,
        graph,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        random_edge=False,
        subchain_length=None,
        equal_partition=True,
        dtype=torch.float32,
    ):
        """
        MPS-based ansatz specific sampler. Will need TN-specific operations.

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
        super().__init__(
            hi,
            graph,
            N_samples,
            burn_in_steps,
            reset_chain,
            random_edge,
            subchain_length,
            equal_partition,
            dtype,
        )

    def _sample_next(self, vstate, burn_in=False):
        """Sample the next configuration. Change the current configuration in place."""
        ind_n_map = {0: 0, 1: 1, 2: 1, 3: 2}

        def exchange_propose(i, j):
            if self.current_config[i] == self.current_config[j]:
                return
            self.attempts += 1
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
                proposed_amp, proposed_amp_tn = vstate.vstate_func.amplitude_n_tn(
                    proposed_config
                )
                proposed_prob = abs(proposed_amp) ** 2
            elif delta_n == 0:
                choices = [(0, 3), (3, 0), (config_j, config_i)]
                choice = random.choice(choices)
                proposed_config[i] = choice[0]
                proposed_config[j] = choice[1]
                proposed_amp, proposed_amp_tn = vstate.vstate_func.amplitude_n_tn(
                    proposed_config
                )
                proposed_prob = abs(proposed_amp) ** 2
            elif delta_n == 2:
                choices = [(config_j, config_i), (1, 2), (2, 1)]
                choice = random.choice(choices)
                proposed_config[i] = choice[0]
                proposed_config[j] = choice[1]
                proposed_amp, proposed_amp_tn = vstate.vstate_func.amplitude_n_tn(
                    proposed_config
                )
                proposed_prob = abs(proposed_amp) ** 2
            else:
                raise ValueError("Invalid configuration")
            try:
                acceptance_ratio = min(1, (proposed_prob / self.current_prob))
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0
            if random.random() < acceptance_ratio or (self.current_prob == 0):
                self.current_config = proposed_config
                self.current_amp = proposed_amp
                self.current_amp_tn = proposed_amp_tn
                self.current_prob = proposed_prob
                vstate.vstate_func.update_cached_cache()
                self.accepts += 1

        site_pairs = self.graph.edges()  # edges sorted from left to right
        with torch.no_grad():
            self.current_amp, self.current_amp_tn = vstate.vstate_func.amplitude_n_tn(
                self.current_config,
                cache="right",
                cache_nn_output=True,
                cache_amp_tn=True,
                initial_cache=True,
            )  # cache env_right
            self.current_prob = abs(self.current_amp) ** 2
            for i, j in site_pairs:
                # t0 = MPI.Wtime()
                exchange_propose(i, j)
                # t1 = MPI.Wtime()
                # t_average += t1-t0
                if i < self.graph.L - 1 and i != 0:
                    vstate.vstate_func.update_env_to_site(
                        self.current_amp_tn, self.current_config, i, from_which="left"
                    )
                # t2 = MPI.Wtime()
                # t_cache += t2-t1
        vstate.vstate_func.clear_env_right_cache()
        # print(f'time for exchange {t_average}, time for caching {t_cache}, average time for caching {t_cache/len(site_pairs)}')

        if burn_in:
            vstate.clear_env_cache()
        else:  # sampling for local energy calculation
            vstate.vstate_func.update_env_to_site(
                self.current_amp_tn, self.current_config, 0, from_which="right"
            )

        if self.current_amp == 0 and DEBUG:
            print(
                f"Rank{RANK}: Warning: psi_sigma is zero for configuration {self.current_config}"
            )

        return self.current_config, self.current_amp


class MetropolisExchangeSamplerSpinful_2D_reusable(Sampler):
    def __init__(
        self,
        hi,
        graph,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        random_edge=False,
        subchain_length=None,
        equal_partition=True,
        dtype=torch.float32,
        hopping_rate=0.5,
    ):
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
        super().__init__(
            hi,
            graph,
            N_samples,
            burn_in_steps,
            reset_chain,
            random_edge,
            subchain_length,
            equal_partition,
            dtype,
        )
        self.hopping_rate = hopping_rate  # Probability of hopping instead of exchanging

    @torch.no_grad()
    def _sample_next(self, vstate, burn_in=False):
        """Sample the next configuration. Change the current configuration in place."""
        ind_n_map = {0: 0, 1: 1, 2: 1, 3: 2}

        def exchange_propose(i, j):
            if self.current_config[i] == self.current_config[j]:
                self.attempts += 1
                self.accepts += 1  # exchange must be accepted, no hopping is allowed
                return
            self.attempts += 1
            proposed_config = self.current_config.clone()
            config_i = self.current_config[i].item()
            config_j = self.current_config[j].item()
            if random.random() < 1 - self.hopping_rate:
                # exchange
                proposed_config[i] = config_j
                proposed_config[j] = config_i
                proposed_amp = vstate.amplitude(proposed_config).cpu()
                proposed_prob = abs(proposed_amp) ** 2
            else:
                # hopping
                n_i = ind_n_map[self.current_config[i].item()]
                n_j = ind_n_map[self.current_config[j].item()]
                delta_n = abs(n_i - n_j)
                if delta_n == 1:
                    # consider only valid hopping: (0, u) -> (u, 0); (d, ud) -> (ud, d)
                    proposed_config[i] = config_j
                    proposed_config[j] = config_i
                    proposed_amp = vstate.amplitude(proposed_config).cpu()
                    proposed_prob = abs(proposed_amp) ** 2
                elif delta_n == 0:
                    # consider only valid hopping: (u, d) -> (0, ud) or (ud, 0)
                    choices = [(0, 3), (3, 0)]
                    choice = random.choice(choices)
                    proposed_config[i] = choice[0]
                    proposed_config[j] = choice[1]
                    proposed_amp = vstate.amplitude(proposed_config).cpu()
                    proposed_prob = abs(proposed_amp) ** 2
                elif delta_n == 2:
                    # consider only valid hopping: (0, ud) -> (u, d) or (d, u)
                    choices = [(1, 2), (2, 1)]
                    choice = random.choice(choices)
                    proposed_config[i] = choice[0]
                    proposed_config[j] = choice[1]
                    proposed_amp = vstate.amplitude(proposed_config).cpu()
                    proposed_prob = abs(proposed_amp) ** 2
                else:
                    raise ValueError("Invalid configuration")
            try:
                acceptance_ratio = min(1, (proposed_prob / self.current_prob))
            except ZeroDivisionError:
                acceptance_ratio = 1 if proposed_prob > 0 else 0
            if random.random() < acceptance_ratio or (self.current_prob == 0):
                self.current_config = proposed_config
                self.current_amp = proposed_amp
                self.current_prob = proposed_prob
                self.accepts += 1

        acceptance_rate = 0

        while acceptance_rate < 0.05:
            self.current_amp = vstate.amplitude(self.current_config).cpu()
            self.current_prob = abs(self.current_amp) ** 2
            proposed_config = self.current_config.clone()

            # initial cache of env_x for current configuration
            vstate.vstate_func.update_env_x_cache_to_row(
                self.current_config, 0, from_which="xmax", mode="force"
            )
            for row_index, row_edges in self.graph.row_edges.items():
                for i, j in row_edges:
                    exchange_propose(i, j)
                if row_index < self.graph.Lx - 1:
                    vstate.vstate_func.update_env_x_cache_to_row(
                        self.current_config, row_index, from_which="xmin", mode="reuse"
                    )
            vstate.vstate_func.clear_env_x_cache()

            vstate.vstate_func.clear_env_y_cache()
            vstate.vstate_func.update_env_y_cache_to_col(
                self.current_config, 1, from_which="ymax", mode="force"
            )
            for col_index, col_edges in self.graph.col_edges.items():
                for i, j in col_edges:
                    exchange_propose(i, j)
                if col_index < self.graph.Ly - 1:
                    vstate.vstate_func.update_env_y_cache_to_col(
                        self.current_config, col_index, from_which="ymin", mode="reuse"
                    )
            vstate.vstate_func.clear_env_y_cache(from_which="ymax")
            vstate.vstate_func.update_env_y_cache_to_col(
                self.current_config, 1, from_which="ymax", mode="force"
            )
            if burn_in:
                vstate.vstate_func.clear_env_y_cache()

            if self.current_amp == 0 and DEBUG:
                print(
                    f"Rank{RANK}: Warning: psi_sigma is zero for configuration {self.current_config}, proposed_config {proposed_config}, proposed_prob {abs(vstate.amplitude(proposed_config))}**2"
                )

            acceptance_rate = self.accepts / self.attempts

            if self.debug:
                print(f"Rank {RANK}: acceptance rate {acceptance_rate} in one sweep")

            if acceptance_rate < 0.05:
                print(f"Rank {RANK}: acceptance rate {acceptance_rate} < 0.05")
                self.reset()
                self.accepts = 0
                self.attempts = 0
                vstate.vstate_func.clear_wavefunction_env_cache()  # Remember to clear the TN contraction cache if need to resample
                if DEBUG:
                    print(f"    Rank {RANK}: acceptance rate {acceptance_rate}")

        return self.current_config, self.current_amp


from pyblock2.driver.core import DMRGDriver, SymmetryTypes  # noqa: E402


class MetropolisMPSSamplerSpinful(Sampler):
    def __init__(
        self,
        hi,
        graph,
        mps_dir="./tmp",
        mps_n_sample=1,
        N_samples=2**8,
        burn_in_steps=100,
        reset_chain=False,
        random_edge=False,
        subchain_length=None,
        equal_partition=False,
        dtype=torch.float32,
    ):
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
        # NOTE: if the MPS distribution differs a lot from the current PEPS distribution, the sampling becomes very inefficient. This may happen for systems with degenerate ground states.
        super().__init__(
            hi,
            graph,
            N_samples,
            burn_in_steps,
            reset_chain,
            random_edge,
            subchain_length,
            equal_partition,
            dtype,
        )
        self.mps_n_sample = mps_n_sample
        self.driver = DMRGDriver(
            scratch=mps_dir, symm_type=SymmetryTypes.SZ, n_threads=1, mpi=True
        )
        self.ket = self.driver.load_mps(tag="KET")

        if self.ket.center != 0:
            if RANK == 0:
                print("Aligning MPS center to 0")
            self.ket = self.driver.copy_mps(self.ket, tag="CSF-TMP")
            self.driver.align_mps_center(self.ket, ref=0)

        print(f"Rank {RANK}: MPS center {self.ket.center}")
        configs, coeffs = self.driver.sample_csf_coefficients(
            self.ket, n_sample=1, iprint=0, rand_seed=RANK + random.randint(0, 2**30)
        )
        self.current_config = configs[0]
        self.current_mps_prob = abs(coeffs[0]) ** 2

    def reset(self):
        configs, coeffs = self.driver.sample_csf_coefficients(
            self.ket, n_sample=1, iprint=0, rand_seed=RANK + random.randint(0, 2**30)
        )
        self.current_config = configs[0]
        self.current_mps_prob = abs(coeffs[0]) ** 2

    def _sample_next(self, vstate, **kwargs):
        """Sample the next configuration. Change the current configuration in place."""
        self.current_amp = vstate.amplitude(self.current_config)
        self.current_prob = abs(self.current_amp) ** 2

        for n_sample in [self.mps_n_sample]:
            configs, coeffs = self.driver.sample_csf_coefficients(
                self.ket,
                n_sample=n_sample,
                iprint=0,
                rand_seed=RANK + random.randint(0, 2**30),
            )
            for proposed_mps_config, proposed_mps_amp in zip(configs, coeffs):
                self.attempts += 1
                proposed_config = proposed_mps_config
                proposed_mps_prob = abs(proposed_mps_amp) ** 2
                proposed_amp = vstate.amplitude(proposed_config)
                proposed_prob = abs(proposed_amp) ** 2
                try:
                    acceptance_ratio = min(
                        1,
                        (proposed_prob / self.current_prob)
                        * (self.current_mps_prob / proposed_mps_prob),
                    )
                except ZeroDivisionError:
                    acceptance_ratio = 1 if proposed_prob > 0 else 0
                # if RANK == 1:
                #     print(f'RANK {RANK}: acceptance ratio {acceptance_ratio}, current_prob {self.current_prob}, current_mps_prob {self.current_mps_prob}, proposed_prob {proposed_prob}, proposed_mps_prob {proposed_mps_prob}')

                if random.random() < acceptance_ratio or (self.current_prob == 0):
                    self.current_config = proposed_config
                    self.current_amp = proposed_amp
                    self.current_prob = proposed_prob
                    self.current_mps_prob = proposed_mps_prob
                    self.accepts += 1
                # if RANK == 1:
                #     print(f'RANK {RANK}: acceptance ratio {acceptance_ratio}, acceptance rate {self.accepts/self.attempts}')

        if self.current_amp == 0 and DEBUG:
            print(
                f"Rank{RANK}: Warning: psi_sigma is zero for configuration {self.current_config}, proposed_config {proposed_config}, proposed_prob {proposed_prob}"
            )

        return self.current_config, self.current_amp
