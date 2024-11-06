import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from mpi4py import MPI
# torch
import torch
# quimb
from autoray import do

from .global_var import DEBUG, set_debug


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


class Variational_State:

    def __init__(self, vstate_func, sampler=None, hi=None, dtype=torch.float32):
        self.vstate_func = vstate_func
        self.sampler = sampler
        self.Np = vstate_func.num_params
        self.hi = sampler.hi if sampler is not None else hi
        self.Ns = sampler.Ns if sampler is not None else self.hi.n_states
        self.equal_partition = sampler.equal_partition
        self.logamp_grad_matrix = None
        self.mean_logamp_grad = None
        self.dtype = dtype
        assert self.hi is not None, "Hilbert space must be provided for sampling!"

    
    @property
    def params_vec(self):
        return self.vstate_func.from_params_to_vec()
    
    @property
    def params_grad_vec(self):
        return self.vstate_func.params_grad_to_vec()
    
    @property
    def model_structure(self):
        try:
            return self.vstate_func.model_structure
        except AttributeError:
            return None
    
    @property
    def state_dict(self):
        return self.vstate_func.state_dict()
    
    # state methods
    
    def reset(self):
        """Clear the gradient of the variational state."""
        self.vstate_func.clear_grad()
    
    def clear_memory(self):
        """Clear the memory of the variational state."""
        self.logamp_grad_matrix = None
        self.mean_logamp_grad = None
        self.reset()

    def update_state(self, new_param_vec):
        """Update the variational state with the new parameter vector."""
        if not type(new_param_vec) == torch.Tensor:
            new_param_vec = torch.tensor(new_param_vec, dtype=self.dtype)
        self.vstate_func.load_params(new_param_vec)
    
    def load_state_dict(self, state_dict):
        self.vstate_func.load_state_dict(state_dict)

    def amplitude(self, x):
        return self.vstate_func(x)
    
    def amplitude_grad(self, x):
        if not type(x) == torch.Tensor:
            x = torch.tensor(np.asarray(x), dtype=self.dtype)
        amp = self.vstate_func(x)
        try:
            amp.backward()
        except RuntimeError:
            # amp is 0
            self.reset()
            return amp, torch.zeros((self.Np,), dtype=self.dtype)
        vec_log_grad = self.vstate_func.params_grad_to_vec()/amp
        # Clear the gradient
        self.reset()
        return amp, vec_log_grad

    def full_hi_logamp_grad_matrix(self):
        """Construct the full Np x Ns matrix of amplitude gradients."""
        parameter_logamp_grad = torch.zeros((self.Np, self.hi.n_states), dtype=self.dtype)
        all_config = self.hi.all_states()
        ampx_arr = torch.zeros((self.hi.n_states,), dtype=self.dtype)

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
        all_config = hi.all_states()
        all_config = np.asarray(all_config)
        psi = self.vstate_func(all_config)
        psi = psi/do('linalg.norm', psi)
        
        op_dense = torch.tensor(op.to_dense(), dtype=self.dtype)
        expect_op = psi.conj().T@(op_dense@psi)

        expect_op.backward()
        op_grad = self.vstate_func.params_grad_to_vec()
        stats_dict = {'mean': float(expect_op.detach().numpy()), 'variance': 0., 'error': 0.}
        # Clear the gradient
        self.reset()
        return stats_dict, op_grad
        
    def expect_and_grad(self, op, full_hi=False, message_tag=None):
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
        
        if self.equal_partition:
            chain_length = self.Ns//SIZE # Number of samples per rank
            # use MPI for sampling
            op_expect, op_grad, op_var, op_err = self.collect_samples(op, chain_length=chain_length)
        
        else:
            assert message_tag is not None, "Message tag must be provided for eager sampling!"
            # use MPI for sampling, but sampler may have different number of samples per rank
            t0 = MPI.Wtime()
            op_expect, op_grad, op_var, op_err = self.collect_samples_eager(op, message_tag=message_tag)
            t1 = MPI.Wtime()
            if RANK == 0:
                print('Time for eager sampling: {}'.format(t1-t0))
        
        # return statistics of the MC sampling
        stats_dict = {'mean': op_expect, 'variance': op_var, 'error': op_err}

        assert self.logamp_grad_matrix is not None, "Logamp gradient matrix must be computed during sampling!"

        return stats_dict, op_grad

    # @profile
    def collect_samples(self, op, chain_length=1):

        vstate = self
        
        # Sample on each rank
        # this should be a list of local samples statistics:
        # a tuple of (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat, W_loc, chain_means_loc)
        t_sample_start = MPI.Wtime()
        local_samples = self.sampler.sample(op, vstate, chain_length=chain_length)
        t_sample_end = MPI.Wtime()
        if DEBUG:
            print('Rank {}, sample time: {}'.format(RANK, t_sample_end-t_sample_start))

        local_op_loc_sum = local_samples[0]
        local_logamp_grad = local_samples[1]
        local_op_logamp_grad_product_sum = local_samples[2]
        local_op_var = local_samples[3]
        local_logamp_grad_matrix = local_samples[4]
        W_loc = local_samples[5]
        chain_means_loc = local_samples[6]
        
        # clean the memory
        del local_samples
        
        t0 = MPI.Wtime()

        # Compute the op expectation value on all ranks
        op_expect = COMM.allreduce(local_op_loc_sum, op=MPI.SUM)/self.Ns # op_expect is seen by all ranks
        mean_logamp_grad = COMM.allreduce(local_logamp_grad, op=MPI.SUM)/self.Ns
        t01 = MPI.Wtime()

        # Collect the op_logamp_grad_product ONLY on rank 0
        op_logamp_grad_product_sum = COMM.reduce(local_op_logamp_grad_product_sum, op=MPI.SUM, root=0)
        t02 = MPI.Wtime()

        # Total sample variance calculation is collected ONLY on rank 0
        local_op_loc_mean = local_op_loc_sum/chain_length
        # (n_i - 1)* s^2_i
        local_op_loc_sqrd_sum = (chain_length-1)*local_op_var + chain_length*(local_op_loc_mean - op_expect)**2
        op_var = COMM.reduce(local_op_loc_sqrd_sum, op=MPI.SUM, root=0)

        # Rhat diagnostics on rank 0. Rhat is the potential scale reduction factor. R\hat = V\hat/W, where V\hat is an unbiased estimation of variance form B and W and W is the within-chain variance.
        # within-chain variance (W)
        W_sum = COMM.reduce(W_loc, op=MPI.SUM, root=0)
        # between-chain variance (B)
        chain_means = COMM.gather(chain_means_loc, root=0) # [[m_11, m_12], [m_21, m_22], ..., [m_q1, m_q2]] where q is the number of ranks

        t1 = MPI.Wtime()

        # set the logamp_grad_matrix and mean_logamp_grad for all ranks
        # each rank has their local batch of logamp_grad_matrix, but shares the same mean_logamp_grad
        self.logamp_grad_matrix = local_logamp_grad_matrix
        self.mean_logamp_grad = mean_logamp_grad

        if RANK == 0:
            print('RANK{}, sample size: {}, chain length per rank: {}'.format(RANK, self.Ns, chain_length))
            print('Time for MPI communication: {}'.format(t1-t0))
            if DEBUG:
                print('     Time for op_expect: {}'.format(t01-t0))
                print('     Time for op_logamp_grad_product_sum: {}'.format(t02-t01))
                print('     Time for op_var: {}'.format(t1-t02))
            op_logamp_grad_product = op_logamp_grad_product_sum/self.Ns
            op_var = op_var/self.Ns

            chain_means = np.concatenate(chain_means, axis=0)
            sub_chain_length = chain_length//2
            sub_chain_num = 2*SIZE
            B = sub_chain_length * np.var(chain_means, ddof=1)
            W = W_sum/sub_chain_num

            # sampling variability correction of the sample mean
            mean_correction = B/(sub_chain_length * sub_chain_num)

            # estimated variance: V\hat
            V_hat = (sub_chain_length - 1)/sub_chain_length * W + B/sub_chain_length# + mean_correction
            R_hat = np.sqrt(V_hat/W)
            print('R_hat: {}'.format(R_hat))
            print('V_hat:{}, OP_var:{}'.format(V_hat, op_var))
            print('logamp_grad_matrix shape: {}'.format(self.logamp_grad_matrix.shape))
            # each logamp_grad is a long vector as (10000,)
            # thus the matrix can be as large as (10000, 10000)

            # Compute the op gradient
            op_grad = op_logamp_grad_product - op_expect*mean_logamp_grad
            # is forming the whole list too memory consuming?

            op_mean_err = np.sqrt(op_var/self.Ns) # standard error of the mean value of the local op

            return op_expect, op_grad, op_var, op_mean_err
        
        else:
            return None, None, None, None
    
    def collect_samples_eager(self, op, message_tag=None):
        vstate = self
        
        # Sample on each rank
        # this should be a list of local samples statistics:
        # a tuple of (op_loc_sum, logpsi_sigma_grad_sum, op_logpsi_sigma_grad_product_sum, op_loc_var, logpsi_sigma_grad_mat)

        t_sample_start = MPI.Wtime()
        local_samples = self.sampler.sample_eager(op, vstate, message_tag=message_tag)
        t_sample_end = MPI.Wtime()
        if DEBUG:
            print('Rank {}, sample time: {}'.format(RANK, t_sample_end-t_sample_start))

        local_op_loc_sum = local_samples[0]
        local_logamp_grad = local_samples[1]
        local_op_logamp_grad_product_sum = local_samples[2]
        local_op_var = local_samples[3]
        local_logamp_grad_matrix = local_samples[4]
        n_local_samples = local_logamp_grad_matrix.shape[1]
        total_sample_Ns = COMM.allreduce(n_local_samples, op=MPI.SUM)

        # clean the memory
        del local_samples
        
        t0 = MPI.Wtime()

        # Compute the op expectation value on all ranks
        op_expect = COMM.allreduce(local_op_loc_sum, op=MPI.SUM)/total_sample_Ns # op_expect is seen by all ranks
        mean_logamp_grad = COMM.allreduce(local_logamp_grad, op=MPI.SUM)/total_sample_Ns
        t01 = MPI.Wtime()

        # Collect the op_logamp_grad_product ONLY on rank 0
        op_logamp_grad_product_sum = COMM.reduce(local_op_logamp_grad_product_sum, op=MPI.SUM, root=0)
        t02 = MPI.Wtime()
        
        # set the logamp_grad_matrix and mean_logamp_grad for all ranks
        # each rank has their local batch of logamp_grad_matrix, but shares the same mean_logamp_grad
        self.logamp_grad_matrix = local_logamp_grad_matrix
        self.mean_logamp_grad = mean_logamp_grad
        

        # compute the total sample variance
        local_op_loc_mean = local_op_loc_sum/n_local_samples
        local_W_var = (n_local_samples-1)*local_op_var + n_local_samples*(local_op_loc_mean - op_expect)**2
        op_var = COMM.reduce(local_W_var, op=MPI.SUM, root=0)

        if RANK == 0:
            print('Total sample size: {}'.format(total_sample_Ns))
            print('Np = {}'.format(self.logamp_grad_matrix.shape[0]))
            if DEBUG:
                print('     Time for op_expect: {}'.format(t01-t0))
                print('     Time for op_logamp_grad_product_sum: {}'.format(t02-t01))
            op_logamp_grad_product = op_logamp_grad_product_sum/total_sample_Ns

            # Compute the op gradient
            op_grad = op_logamp_grad_product - op_expect*mean_logamp_grad

            op_var = op_var/total_sample_Ns
            op_mean_err = np.sqrt(op_var/total_sample_Ns)

            return op_expect, op_grad, op_var, op_mean_err
        
        else:
            return None, None, None, None
            

