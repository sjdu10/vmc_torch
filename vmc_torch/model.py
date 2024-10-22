from mpi4py import MPI
import ast
# torch
import torch
import torch.nn as nn

# quimb
import quimb.tensor as qtn
import symmray as sr
import autoray as ar
from autoray import do

from .fermion_utils import insert_proj_peps, flatten_proj_params, reconstruct_proj_params
from .global_var import DEBUG, set_debug

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def init_weights_xavier(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Define the custom Kaiming initialization function
def init_weights_kaiming(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # Initialize weights using Kaiming uniform
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Define the custom zero-initialization function
def init_weights_to_zero(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
        # Set weights and biases to zero
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, 0.0)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)



class SlaterDeterminant(nn.Module):
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32):
        super(SlaterDeterminant, self).__init__()
        
        self.hilbert = hilbert
        self.param_dtype = param_dtype
        
        # Initialize the parameter M (N x Nf matrix)
        self.M = nn.Parameter(
            kernel_init(torch.empty(self.hilbert.n_orbitals, self.hilbert.n_fermions, dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.n_orbitals, self.hilbert.n_fermions, dtype=self.param_dtype)
        )

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'SlaterDeterminant':{'N_orbitals': self.hilbert.n_orbitals, 'N_fermions': self.hilbert.n_fermions}
        }
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    def _determinant(self, A):
        # Compute the determinant of matrix A
        det = torch.linalg.det(A)
        return det

    def forward(self, n):
        if not type(n) == torch.Tensor:
            n = torch.tensor(n, dtype=torch.int32)
        # Define the slater determinant function manually to loop over inputs
        def slater_det(n):
            # Find the positions of the occupied orbitals
            R = torch.nonzero(n, as_tuple=False).squeeze()

            # Extract the Nf x Nf submatrix of M corresponding to the occupied orbitals
            A = self.M[R]

            return self._determinant(A)
        if n.ndim == 1:
            # If input is not batched, add a batch dimension
            n = n.unsqueeze(0)
        # Apply slater_det to each element in the batch
        batch_size = n.shape[0]
        return torch.stack([slater_det(n[i]) for i in range(batch_size)])

class NeuralJastrow(nn.Module):
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64):
        super(NeuralJastrow, self).__init__()
        
        self.hilbert = hilbert
        self.param_dtype = param_dtype
        self.hidden_dim = hidden_dim
        
        # Initialize the parameter M (N x Nf matrix)
        self.M = nn.Parameter(
            kernel_init(torch.empty(self.hilbert.n_orbitals, self.hilbert.n_fermions, dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.n_orbitals, self.hilbert.n_fermions, dtype=self.param_dtype)
        )

        self.nn = nn.Sequential(
            nn.Linear(self.hilbert.n_orbitals, self.hidden_dim, dtype=self.param_dtype),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1, dtype=self.param_dtype)
        )

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'NeuralJastrow':{'N_orbitals': self.hilbert.n_orbitals, 'N_fermions': self.hilbert.n_fermions}
        }
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    def _determinant(self, A):
        # Compute the determinant of matrix A
        det = torch.linalg.det(A)
        return det

    def forward(self, n):
        if not type(n) == torch.Tensor:
            n = torch.tensor(n, dtype=torch.int32)
        n = n.to(self.param_dtype)
        # Define the slater determinant function manually to loop over inputs
        def slater_det_Jastrow(n):
            # Find the positions of the occupied orbitals
            R = torch.nonzero(n, as_tuple=False).squeeze()

            # Extract the Nf x Nf submatrix of M corresponding to the occupied orbitals
            A = self.M[R]

            # Jastrow factor
            J = torch.sum(self.nn(n))

            return self._determinant(A)*torch.exp(J)
        
        if n.ndim == 1:
            # If input is not batched, add a batch dimension
            n = n.unsqueeze(0)
        # Apply slater_det to each element in the batch
        batch_size = n.shape[0]
        return torch.stack([slater_det_Jastrow(n[i]) for i in range(batch_size)])

class NeuralBackflow(nn.Module):
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64):
        super(NeuralBackflow, self).__init__()
        
        self.hilbert = hilbert
        self.param_dtype = param_dtype
        
        # Initialize the parameter M (N x Nf matrix)
        self.M = nn.Parameter(
            kernel_init(torch.empty(self.hilbert.n_orbitals, self.hilbert.n_fermions, dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.n_orbitals, self.hilbert.n_fermions, dtype=self.param_dtype)
        )

        # Initialize the neural network layer, input is n and output a matrix with the same shape as M
        self.nn = nn.Sequential(
            nn.Linear(self.hilbert.n_orbitals, hidden_dim, dtype=self.param_dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.hilbert.n_orbitals*self.hilbert.n_fermions, dtype=self.param_dtype)
        )

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'Neuralbackflow':{'N_orbitals': self.hilbert.n_orbitals, 'N_fermions': self.hilbert.n_fermions}
        }
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    

    def forward(self, n):
        if not type(n) == torch.Tensor:
            n = torch.tensor(n, dtype=self.param_dtype)
        n = n.to(self.param_dtype)
        # Define the slater determinant function manually to loop over inputs
        def backflow_det(n):
            # Compute the backflow matrix F using the neural network
            F = self.nn(n)
            M  = self.M + F.reshape(self.M.shape)
            # Find the positions of the occupied orbitals
            R = torch.nonzero(n, as_tuple=False).squeeze()

            # Extract the Nf x Nf submatrix of M corresponding to the occupied orbitals
            A = M[R]

            det = torch.linalg.det(A)
            return det

        if n.ndim == 1:
            # If input is not batched, add a batch dimension
            n = n.unsqueeze(0)
        # Apply slater_det to each element in the batch
        batch_size = n.shape[0]
        return torch.stack([backflow_det(n[i]) for i in range(batch_size)])



class FFNN(nn.Module):
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64):
        super(FFNN, self).__init__()
        
        self.hilbert = hilbert
        self.param_dtype = param_dtype
        
        # Initialize the parameter M (N x Nf matrix)
        self.M = nn.Parameter(
            kernel_init(torch.empty(self.hilbert.n_orbitals, self.hilbert.n_fermions, dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.n_orbitals, self.hilbert.n_fermions, dtype=self.param_dtype)
        )

        # Initialize the neural network layer, input is n and output a matrix with the same shape as M
        self.nn = nn.Sequential(
            nn.Linear(self.hilbert.n_orbitals, hidden_dim, dtype=self.param_dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, dtype=self.param_dtype)
        )

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'FFNN':{'N_orbitals': self.hilbert.n_orbitals, 'N_fermions': self.hilbert.n_fermions}
        }
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param

    def forward(self, n):
        if not type(n) == torch.Tensor:
            n = torch.tensor(n, dtype=self.param_dtype)
        # Define the slater determinant function manually to loop over inputs
        def ffnn(n):
            # Compute the backflow matrix F using the neural network
            F = sum(self.nn(n))
            return F
        if n.ndim == 1:
            # If input is not batched, add a batch dimension
            n = n.unsqueeze(0)
        # Apply slater_det to each element in the batch
        batch_size = n.shape[0]
        return torch.stack([ffnn(n[i]) for i in range(batch_size)])