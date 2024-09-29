import numpy as np
from mpi4py import MPI
# scipy
import scipy
from scipy.sparse import csr_matrix

# torch
import torch

# quimb
from autoray import do


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


class Preconditioner:
    def __init__(self, use_MPI4Solver=False):
        self.use_MPI4Solver = use_MPI4Solver
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
    def __init__(self, dense=False, exact=False, iter_step=1e3, use_MPI4Solver=False):
        super().__init__(use_MPI4Solver)
        self.dense = dense
        self.iter_step = iter_step
        self.exact = exact
    def __call__(self, state, energy_grad, eta=1e-3):
        """iter_step is for iterative solvers."""
        
        if self.exact:
            if energy_grad is None:
                return torch.zeros(state.Np, dtype=torch.float32)
            parameter_amp_grad, amp_arr = state.get_logamp_grad_matrix()
            parameter_amp_grad = parameter_amp_grad.detach().numpy()
            amp_arr = amp_arr.detach().numpy()
            norm_sqr = np.linalg.norm(amp_arr)**2
            S = np.sum([np.outer(amp_grad, amp_grad.conj()) for amp_grad in parameter_amp_grad.T], axis=0)/norm_sqr
            weighted_amp_grad = np.sum([amp_arr[i]*parameter_amp_grad[:, i] for i in range(amp_arr.shape[0])], axis=0)/norm_sqr
            S -= np.outer(weighted_amp_grad, weighted_amp_grad.conj())
            R = S + eta*np.eye(S.shape[0])
            dp = scipy.linalg.solve(R, energy_grad.detach().numpy())
            return torch.tensor(dp, dtype=torch.float32)

        if self.dense:
            logamp_grad_matrix, mean_logamp_grad = state.get_logamp_grad_matrix()
            logamp_grad_matrix_list = COMM.gather(local_logamp_grad_matrix, root=0)
            if energy_grad is None:
                return torch.zeros(state.Np, dtype=torch.float32)
            logamp_grad_matrix = np.concatenate(logamp_grad_matrix_list, axis=1)
            if type(logamp_grad_matrix) is torch.Tensor:
                logamp_grad_matrix = logamp_grad_matrix.detach().numpy()
            # form the dense S matrix
            S = np.mean([np.outer(logamp_grad, logamp_grad.conj()) for logamp_grad in logamp_grad_matrix.T], axis=0)
            S -= np.outer(mean_logamp_grad, mean_logamp_grad.conj())
            R = S + eta*np.eye(S.shape[0])
            R = csr_matrix(R)
            # dp = scipy.linalg.solve(R, energy_grad)
            dp = scipy.sparse.linalg.cg(R, energy_grad)[0]

            return torch.tensor(dp, dtype=torch.float32)
        
        else:
            if self.use_MPI4Solver:
                """Solve the linear equation locally in each MPI rank.
                Save the step of gathering the logamp_grad_matrix to rank 0, which can be slow.
                This is at the cost of having to broadcast the energy_grad to all ranks
                and solve the linear equation locally in every rank."""
                local_logamp_grad_matrix, mean_logamp_grad = state.get_logamp_grad_matrix()
                energy_grad = COMM.bcast(energy_grad, root=0)
                def R_dot_x(x, eta=1e-6):
                    x_out_local = np.zeros_like(x)
                    for i in range(local_logamp_grad_matrix.shape[1]):
                        x_out_local += np.dot(local_logamp_grad_matrix[:, i], x)*local_logamp_grad_matrix[:, i]
                    x_out = COMM.allreduce(x_out_local, op=MPI.SUM)
                    x_out -= np.dot(mean_logamp_grad, x)*mean_logamp_grad
                    return x_out + eta*x
                n = state.Np
                matvec = lambda x: R_dot_x(x, eta)
                A = scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec)
                b = energy_grad.detach().numpy() if type(energy_grad) is torch.Tensor else energy_grad
                dp, _ = scipy.sparse.linalg.cg(A, b)
                return torch.tensor(dp, dtype=torch.float32)


            else:
                """Solve the linear equation in rank 0 after sending all logamp_grad vectors to rank 0 from all ranks."""
                local_logamp_grad_matrix, mean_logamp_grad = state.get_logamp_grad_matrix()
                logamp_grad_matrix_list = COMM.gather(local_logamp_grad_matrix, root=0)
                if energy_grad is None:
                    # All ranks but rank 0 return zeros
                    return torch.zeros(state.Np, dtype=torch.float32)
                logamp_grad_matrix = np.concatenate(logamp_grad_matrix_list, axis=1)
                # define function of (S+eta*I) dot x
                def R_dot_x(x, logamp_grad_matrix, mean_logamp_grad, eta=1e-6):
                    x_out = np.zeros_like(x)
                    for i in range(logamp_grad_matrix.shape[1]):
                        x_out += np.dot(logamp_grad_matrix[:, i], x)*logamp_grad_matrix[:, i]
                    x_out -= np.dot(mean_logamp_grad, x)*mean_logamp_grad
                    return x_out + eta*x 
                
                # define the linear operator
                n = state.Np
                matvec = lambda x: R_dot_x(x, logamp_grad_matrix, mean_logamp_grad, eta)
                A = scipy.sparse.linalg.LinearOperator((n, n), matvec=matvec)
                # Right-hand side vector
                b = energy_grad.detach().numpy() if type(energy_grad) is torch.Tensor else energy_grad
                # Solve the linear equation
                dp, _ = scipy.sparse.linalg.cg(A, b)
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