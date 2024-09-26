import numpy as np
from mpi4py import MPI
# torch
import torch
# quimb
from autoray import do

from global_var import DEBUG, set_debug


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


class Variational_State:

    def __init__(self, vstate_func, sampler=None, hi=None):
        self.vstate_func = vstate_func
        self.sampler = sampler
        self.Np = vstate_func.num_params
        self.hi = sampler.hi if sampler is not None else hi
        self.Ns = sampler.Ns if sampler is not None else self.hi.n_states
        self.nsites = self.hi.size
        self.logamp_grad_matrix = None
        self.mean_logamp_grad = None
        assert self.hi is not None, "Hilbert space must be provided for sampling!"

    
    def reset(self):
        """Clear the gradient of the variational state."""
        self.vstate_func.clear_grad()

    def update_state(self, new_param_vec):
        """Update the variational state with the new parameter vector."""
        if not type(new_param_vec) == torch.Tensor:
            new_param_vec = torch.tensor(new_param_vec, dtype=torch.float32)
        self.vstate_func.load_params(new_param_vec)

    @property
    def params_vec(self):
        return self.vstate_func.from_params_to_vec()
    
    @property
    def params_grad_vec(self):
        return self.vstate_func.params_grad_to_vec()
    
    def amplitude(self, x):
        return self.vstate_func(x)
    
    def amplitude_grad(self, x):
        if not type(x) == torch.Tensor:
            x = torch.tensor(np.asarray(x), dtype=torch.float32)
        amp = self.vstate_func(x)
        amp.backward()
        vec_log_grad = self.vstate_func.params_grad_to_vec()/amp
        # Clear the gradient
        self.reset()
        return amp, vec_log_grad

    def full_hi_logamp_grad_matrix(self):
        """Construct the full Np x Ns matrix of amplitude gradients."""
        parameter_logamp_grad = torch.zeros((self.Np, self.hi.n_states), dtype=torch.float32)
        all_config = self.hi.all_states()
        ampx_arr = torch.zeros((self.hi.n_states,), dtype=torch.float32)

        for i, config in enumerate(all_config):
            ampx, ampx_dp = self.amplitude_grad(config)
            parameter_logamp_grad[:, i] = ampx_dp
            ampx_arr[i] = ampx

        return parameter_logamp_grad, ampx_arr
    

    def get_logamp_grad_matrix(self):
        """Return the amplitude gradient matrix."""
        if self.logamp_grad_matrix is None:
            if self.sampler is None and self.hi is not None:
                return self.full_hi_logamp_grad_matrix()
            else:
                raise ValueError("Sampler must be provided for sampling!")
        else:
            # should be computed during sampling
            return self.logamp_grad_matrix, self.mean_logamp_grad
    

    def full_hi_expect_and_grad(self, op):
        """Full Hilbert space expectation value and gradient calculation.
        Only for sanity check on small systems.
        """
        hi = op.hilbert # netket hilbert object
        N = hi.size
        all_config = hi.all_states()
        psi = self.vstate_func(all_config)
        psi = psi/do('linalg.norm', psi)
        
        op_dense = torch.tensor(op.to_dense(), dtype=torch.float32)
        expect_op_per_site = psi.conj().T@(op_dense@psi)/N

        expect_op_per_site.backward()
        vec_grad = self.vstate_func.params_grad_to_vec()
        # Clear the gradient
        self.reset()
        return expect_op_per_site, vec_grad
        
    
    def expect_and_grad(self, op, full_hi=False):
        """
        Compute the expectation value of the operator `op` and its gradient.

        Args:
            op (netket operator object): The operator for which the expectation value and gradient are calculated.
            full_hi (bool): Whether to use the full Hilbert space expectation value and gradient calculation.

        Returns:
            torch.tensor: The expectation value of the operator.
            torch.tensor: The gradient of the expectation value with respect to the parameters.
        """
        if full_hi or self.sampler is None:
            return self.full_hi_expect_and_grad(op)
        
        # use MPI for sampling
        chain_length = self.Ns//SIZE # Number of samples per rank

        op_expect, op_grad, op_var, op_err = self.collect_samples(op, chain_length=chain_length)
        
        # return statistics of the MC sampling
        stats_dict = {'mean': op_expect, 'variance': op_var, 'error': op_err}


        return stats_dict, op_grad

    
    def collect_samples(self, op, chain_length=1):

        vstate = self
        
        # Sample on each rank
        # this should be a list of local samples statistics:
        # a tuple of (op_loc_sum, logamp_grad_sum, op_loc_logamp_grad_product_sum, op_loc_var, logamp_grad_matrix)
        local_samples = self.sampler.sample(op, vstate, chain_length=chain_length)

        local_op_loc = local_samples[0]
        local_logamp_grad = local_samples[1]
        local_op_logamp_grad_product = local_samples[2]
        local_op_var = local_samples[3]
        local_logamp_grad_matrix = local_samples[4]
        
        # clean the memory
        del local_samples
        
        # Compute the op expectation value on all ranks
        op_expect = COMM.allreduce(local_op_loc, op=MPI.SUM)/self.Ns # op_expect is seen by all ranks
        mean_logamp_grad = COMM.allreduce(local_logamp_grad, op=MPI.SUM)/self.Ns

        # Collect the op_logamp_grad_product on rank 0
        op_logamp_grad_product = COMM.reduce(local_op_logamp_grad_product, op=MPI.SUM, root=0)

        # Total sample variance calculation is collected on rank 0
        local_op_loc_mean = local_op_loc/chain_length
        # (n_i - 1)* s^2_i
        local_op_loc_sqrd_sum = (chain_length-1)*local_op_var + chain_length*(local_op_loc_mean - op_expect)**2
        op_var = COMM.reduce(local_op_loc_sqrd_sum, op=MPI.SUM, root=0)

        # set the logamp_grad_matrix and mean_logamp_grad for all ranks
        # each rank has their local batch of logamp_grad_matrix, but shares the same mean_logamp_grad
        self.logamp_grad_matrix = local_logamp_grad_matrix
        self.mean_logamp_grad = mean_logamp_grad

        # reset sampler
        self.sampler.reset()

        if RANK == 0:

            print('RANK{}, sample size: {}, chain length per rank: {}'.format(RANK, self.Ns, chain_length))
            op_logamp_grad_product = op_logamp_grad_product/self.Ns
            op_var = op_var/self.Ns
            # Join all samples list from all ranks into a single list
            # each sample is a tuple of (config, E_loc, amp, logamp_grad)
            # logamp_grad_matrix = np.concatenate(logamp_grad_matrix_list, axis=1)
            # del logamp_grad_matrix_list
            # self.logamp_grad_matrix = logamp_grad_matrix # Only rank 0 has the full logamp_grad_matrix

            if DEBUG:
                print('logamp_grad_matrix shape: {}'.format(self.logamp_grad_matrix.shape))
                # each logamp_grad is a long vector as (10000,)
                # thus the matrix can be as large as (10000, 10000)

            # Compute the op gradient
            op_grad = op_logamp_grad_product - op_expect*mean_logamp_grad
            # is forming the whole list too memory consuming?

            op_err = np.sqrt(op_var/self.Ns)

            return op_expect, op_grad, op_var, op_err
        
        else:
            return None, None, None, None
        

            

