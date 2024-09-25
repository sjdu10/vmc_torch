import numpy as np
from mpi4py import MPI
import scipy

# torch
import torch

# quimb
from autoray import do


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


class Preconditioner:
    def __call__(self, state, grad):
        """Abstract method for preconditioning the gradient."""
        
        raise NotImplementedError

class TrivialPreconditioner(Preconditioner):
    """Trivial preconditioner that does nothing."""
    def __call__(self, state, grad):
        return grad

class SR(Preconditioner):
    """
    Math: S*dp = g, where S is the QGT and g is the energy gradient. dp is the preconditioned gradient.
    g = <E_loc(x)*O(x)> - <E_loc(x)>*<O(x)>, where O(x) = \nabla_{\theta} log(psi(x;\theta)) is the gradient of the log amplitude.
    O(x) has shape of (Np,), where Np is the number of parameters.

    S = <O^\dagger(x)*O(x)> - <O^\dagger(x)>*<O(x)>, which has shape of (Np, Np).
    S is a positive definite matrix.
    S can be computed from the amp_grad matrix, which has shape of (Np, Ns), where Ns is the number of samples.
    S is just the covariance matrix of the amp_grad vectors.

    In practice, one does not need to compute the dense S matrix to solve for dp.
    One can solve the linear equation S*dp = g iteratively using scipy.sparse.linalg.
    """
    def __init__(self, dense=False, iter_step=None, exact=False):
        self.dense = dense
        self.iter_step = iter_step
        self.exact = exact
    def __call__(self, state, energy_grad, eta=1e-3):
        """iter_step is for iterative solvers."""
        if self.exact:
            parameter_amp_grad, amp_arr = state.get_amp_grad_matrix()
            parameter_amp_grad = parameter_amp_grad.detach().numpy()
            amp_arr = amp_arr.detach().numpy()
            norm_sqr = np.linalg.norm(amp_arr)**2
            S = np.sum([np.outer(amp_grad, amp_grad.conj()) for amp_grad in parameter_amp_grad.T], axis=0)/norm_sqr
            weighted_amp_grad = np.sum([amp_arr[i]*parameter_amp_grad[:, i] for i in range(amp_arr.shape[0])], axis=0)/norm_sqr
            S -= np.outer(weighted_amp_grad, weighted_amp_grad.conj())
            R = S + eta*np.eye(S.shape[0])
            dp = scipy.linalg.solve(R, energy_grad.detach().numpy())
            return torch.tensor(dp, dtype=torch.float32)

        amp_grad_matrix_normalized = state.get_amp_grad_matrix()
        if type(amp_grad_matrix_normalized) is torch.Tensor:
            amp_grad_matrix_normalized = amp_grad_matrix_normalized.detach().numpy()
        if self.dense:
            # form the dense S matrix
            S = np.mean([np.outer(amp_grad, amp_grad.conj()) for amp_grad in amp_grad_matrix_normalized.T], axis=0)
            S -= amp_grad_matrix_normalized.mean(axis=1)@amp_grad_matrix_normalized.mean(axis=1).T
            R = S + eta*np.eye(S.shape[0])
            dp = scipy.linalg.solve(R, energy_grad)
            return torch.tensor(dp, dtype=torch.float32)



class Optimizer:
    def __init__(self, learning_rate=1e-3):
        self.lr = learning_rate
    
    def compute_update_params(self, params, grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate)
    
    def compute_update_params(self, params, grad):
        return params - self.lr*grad

class SignedSGD(Optimizer):
    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate)
    
    def compute_update_params(self, params, grad):
        if type(grad) != torch.Tensor:
            grad = torch.tensor(grad, dtype=torch.float32)
        return params - self.lr*torch.sign(grad)

class SignedRandomSGD(Optimizer):
    def __init__(self, learning_rate=1e-3):
        super().__init__(learning_rate)
    
    def compute_update_params(self, params, grad):
        return params - self.lr*torch.sign(grad)*torch.rand(1)