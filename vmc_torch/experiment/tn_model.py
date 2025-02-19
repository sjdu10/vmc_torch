from mpi4py import MPI
import ast
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# quimb
import quimb.tensor as qtn
from quimb.tensor.tensor_2d import Rotator2D, pairwise
import symmray as sr
import autoray as ar
from autoray import do
import cotengra as ctg

from vmc_torch.fermion_utils import insert_proj_peps, flatten_proj_params, reconstruct_proj_params, insert_compressor
from vmc_torch.global_var import DEBUG, set_debug
flatten_tn_params = flatten_proj_params

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
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Define the custom zero-initialization function
def init_weights_to_zero(m, std=1e-3):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
        # Set weights and biases to zero
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, mean=0.0, std=std)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Define the custom uniform initialization function
def init_weights_uniform(m, a=-5e-3, b=5e-3):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
        # Set weights and biases to zero
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.uniform_(m.weight, a=a, b=b)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.uniform_(m.bias, a=a, b=b)


#------------ bosonic TN model ------------


class PEPS_model(torch.nn.Module):
    def __init__(self, peps, max_bond=None):
        super().__init__()
        self.peps = peps
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.model_structure = {
            'PEPS':{'D': peps.max_bond(), 'Lx': peps.Lx, 'Ly': peps.Ly},
        }
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(peps)

        self.torch_tn_params = {
            str(tid): nn.Parameter(data)  # Convert Tensor to Parameter
            for tid, data in params.items()
        }
        # register the torch tensors as parameters
        for tid, param in self.torch_tn_params.items():
            self.register_parameter(tid, param)
        
        # self.load_params(self.from_params_to_vec())
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    def load_params(self, vec):
        pointer = 0
        for param in self.parameters():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    def from_vec_to_params(self, vec):
        # XXX: useful at all?
        pointer = 0
        new_params = {}
        for tid, param in self.torch_tn_params.items():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            new_params[tid] = new_param_values
            pointer += num_param
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def amplitude(self, x):
        # update self.PEPS
        params ={
            int(tid): data
            for tid, data in self.torch_tn_params.items()
        }
        peps = qtn.unpack(params, self.skeleton)
        def func(xi):
            if self.max_bond is None:
                amp_val = peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(xi)}).contract()
                return amp_val
            else:
                amp = peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(xi)})
                amp.contract_boundary_from_ymin_(max_bond=self.max_bond, cutoff=0.0, yrange=[0, peps.Ly//2-1],canonize=True)
                amp.contract_boundary_from_ymax_(max_bond=self.max_bond, cutoff=0.0, yrange=[peps.Ly//2, peps.Ly-1],canonize=True)
                amp_val = amp.contract()
                return amp_val
        
        if x.ndim == 1:
            return func(x)
        else:
            return torch.stack([func(xi) for xi in x])
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)

class PEPS_NN_Model(torch.nn.Module):
    def __init__(self, peps, max_bond=None, nn_hidden_dim=64, nn_eta=1e-3, activation='ReLU', param_dtype=torch.float32):
        super().__init__()
        self.peps = peps
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.param_dtype = param_dtype
        self.model_structure = {
            'PEPS':{'D': peps.max_bond(), 'Lx': peps.Lx, 'Ly': peps.Ly, 'nn_hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'max_bond': max_bond, 'NN': '2LayerMLP', 'Activation': activation},
        }
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(peps)

        self.torch_tn_params = {
            str(tid): nn.Parameter(data)  # Convert Tensor to Parameter
            for tid, data in params.items()
        }
        # register the torch tensors as parameters
        for tid, param in self.torch_tn_params.items():
            self.register_parameter(tid, param)
        
        dummy_config = torch.zeros(peps.nsites)
        dummy_amp_2row = peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(dummy_config)}).contract_boundary_from_ymin(max_bond=max_bond, cutoff=0.0, yrange=[0, peps.Ly//2-1])
        dummy_amp_2row = dummy_amp_2row.contract_boundary_from_ymax(max_bond=max_bond, cutoff=0.0, yrange=[peps.Ly//2, peps.Ly-1])
        dummy_2row_params, dummy_2row_skeleton = qtn.pack(dummy_amp_2row)

        tworow_params_vec = self.from_two_row_params_to_vec(dummy_2row_params)
        self.tworow_params_vec_len = len(tworow_params_vec)
        self.tworow_params_example = dummy_2row_params

        # Define an MLP layer (or any other neural network layers)
        activation_func = getattr(nn, activation)
        self.mlp = nn.Sequential(
            nn.Linear(peps.nsites, nn_hidden_dim),
            activation_func(),
            nn.Linear(nn_hidden_dim, self.tworow_params_vec_len)
        )
        self.nn_eta = nn_eta
    
    def from_two_row_params_to_vec(self, two_row_params):
        return torch.cat([param.flatten() for param in two_row_params.values()])
    
    def from_vec_to_two_row_params(self, vec):
        pointer = 0
        new_two_row_params = {}
        for tid, param in self.tworow_params_example.items():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            new_two_row_params[tid] = new_param_values
            pointer += num_param
        return new_two_row_params
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    def load_params(self, vec):
        pointer = 0
        for param in self.parameters():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)
    
    def amplitude(self, x):
        params ={
            int(tid): data
            for tid, data in self.torch_tn_params.items()
        }
        peps = qtn.unpack(params, self.skeleton)
        def func(xi):
            tworow_tn = peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(xi)}).contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, peps.Ly//2-1])
            tworow_tn = tworow_tn.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[peps.Ly//2, peps.Ly-1])
            tworow_params, tworow_skeleton = qtn.pack(tworow_tn)
            tworow_params_vec = self.from_two_row_params_to_vec(tworow_params)
            if type(xi) is not torch.Tensor:
                xi = torch.tensor(xi, dtype=self.param_dtype)
            tworow_params_vec = tworow_params_vec + self.mlp(xi)*self.nn_eta
            new_tworow_params = self.from_vec_to_two_row_params(tworow_params_vec)
            new_tworow_tn = qtn.unpack(new_tworow_params, tworow_skeleton)
            return new_tworow_tn.contract()
        
        if x.ndim == 1:
            return func(x)
        else:
            return torch.stack([func(xi) for xi in x])


class PEPS_NNproj_Model(torch.nn.Module):
    def __init__(self, peps, max_bond=None, nn_hidden_dim=64, nn_eta=1e-3, activation='ReLU', param_dtype=torch.float32):
        super().__init__()
        self.peps = peps
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.param_dtype = param_dtype
        self.model_structure = {
            'PEPS':{'D': peps.max_bond(), 'Lx': peps.Lx, 'Ly': peps.Ly, 'nn_hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'max_bond': max_bond, 'NN': '2LayerMLP', 'Activation': activation},
        }
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(peps)

        self.torch_tn_params = {
            str(tid): nn.Parameter(data)  # Convert Tensor to Parameter
            for tid, data in params.items()
        }
        # register the torch tensors as parameters
        for tid, param in self.torch_tn_params.items():
            self.register_parameter(tid, param)
        
        dummy_config = torch.zeros(peps.nsites)
        dummy_tn_amp = peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(dummy_config)})
        dummy_tn_w_proj = dummy_tn_amp.contract_boundary_from(
            xrange=None, 
            yrange=(0,peps.Ly-2), 
            max_bond=self.max_bond,
            cutoff=0.0, 
            from_which='ymin',
            mode='projector2d',
            lazy=True, 
            new_tags=['proj'],
            )
        dummy_tn, proj_tn = dummy_tn_w_proj.partition(tags='proj')
        proj_params, proj_skeleton = qtn.pack(proj_tn)
        proj_params_vec = self.from_tn_params_to_vec(proj_params)
        self.proj_params_vec_len = len(proj_params_vec)
        self.proj_params_example = proj_params

        # Define an MLP layer (or any other neural network layers)
        activation_func = getattr(nn, activation)
        self.mlp = nn.Sequential(
            nn.Linear(peps.nsites, nn_hidden_dim),
            activation_func(),
            nn.Linear(nn_hidden_dim, self.proj_params_vec_len)
        )
        self.nn_eta = nn_eta
    
    def from_tn_params_to_vec(self, tn_params):
        return torch.cat([param.flatten() for param in tn_params.values()])
    
    def from_vec_to_tn_params(self, vec, tn_params_example):
        pointer = 0
        new_tn_params = {}
        for tid, param in tn_params_example.items():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            new_tn_params[tid] = new_param_values
            pointer += num_param
        return new_tn_params
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    def load_params(self, vec):
        pointer = 0
        for param in self.parameters():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)
    
    def amplitude(self, x):
        params ={
            int(tid): data
            for tid, data in self.torch_tn_params.items()
        }

        peps = qtn.unpack(params, self.skeleton)
        def func(xi):
            tn_amp = peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(xi)})
            tn_w_proj = tn_amp.contract_boundary_from(
                xrange=None,
                yrange=(0,peps.Ly-2),
                max_bond=self.max_bond,
                cutoff=0.0,
                from_which='ymin',
                mode='projector2d',
                lazy=True,
                new_tags=['proj'],
            )
            tn, proj_tn = tn_w_proj.partition(tags='proj')
            proj_params, proj_skeleton = qtn.pack(proj_tn)
            proj_params_vec = self.from_tn_params_to_vec(proj_params)
            if type(xi) is not torch.Tensor:
                xi = torch.tensor(xi, dtype=self.param_dtype)
            proj_params_vec = proj_params_vec + self.mlp(xi)*self.nn_eta
            new_proj_params = self.from_vec_to_tn_params(proj_params_vec, self.proj_params_example)
            new_proj_tn = qtn.unpack(new_proj_params, proj_skeleton)
            new_tn_w_proj = tn | new_proj_tn
            return new_tn_w_proj.contract()
        
        if x.ndim == 1:
            return func(x)
        else:
            return torch.stack([func(xi) for xi in x])

class PEPS_delocalized_Model(torch.nn.Module):
    def __init__(self, peps, max_bond=None, diag=False):
        super().__init__()
        dpeps = self.delocalize_in_cross(peps, diag=diag)
        self.peps = dpeps
        self.diag = diag
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.model_structure = {
            'PEPS_delocalized':{'D': self.peps.max_bond(), 'Lx': self.peps.Lx, 'Ly': self.peps.Ly, 'diag': diag},
        }
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(self.peps)

        self.torch_tn_params = {
            str(tid): nn.Parameter(data)  # Convert Tensor to Parameter
            for tid, data in params.items()
        }
        # register the torch tensors as parameters
        for tid, param in self.torch_tn_params.items():
            self.register_parameter(tid, param)
        
    
    def delocalize_in_cross(self, peps, diag=False):

        peps = peps.copy()

        for (i, j) in peps.sites:
            t = peps[i, j]
            if diag:
                neighbors = [
                    (i + 1, j + 1),
                    (i - 1, j - 1),
                    (i + 1, j - 1),
                    (i - 1, j + 1),
                ]
            else:
                neighbors = [
                    (i + 1, j),
                    (i - 1, j),
                    (i, j + 1),
                    (i, j - 1),
                ]
            
            for neighbor in neighbors:
                if peps.valid_coo(neighbor):
                    t.new_ind(peps.site_ind(neighbor), mode="repeat", size=2)

        return peps
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    def load_params(self, vec):
        pointer = 0
        for param in self.parameters():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    def from_vec_to_params(self, vec):
        # XXX: useful at all?
        pointer = 0
        new_params = {}
        for tid, param in self.torch_tn_params.items():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            new_params[tid] = new_param_values
            pointer += num_param
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def amplitude(self, x):
        # update self.PEPS
        params ={
            int(tid): data
            for tid, data in self.torch_tn_params.items()
        }
        peps = qtn.unpack(params, self.skeleton)
        def func(xi):
            if self.max_bond is None:
                return peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(xi)}).contract()
            else:
                amp = peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(xi)})
                amp.contract_boundary_from_ymin_(max_bond=self.max_bond, cutoff=0.0, yrange=[0, peps.Ly//2-1])
                amp.contract_boundary_from_ymax_(max_bond=self.max_bond, cutoff=0.0, yrange=[peps.Ly//2, peps.Ly-1])
                return amp.contract()
        
        if x.ndim == 1:
            return func(x)
        else:
            return torch.stack([func(xi) for xi in x])
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)

#------------ fermionic TN model ------------

class wavefunctionModel(torch.nn.Module):
    """Common class functions for all VMC models"""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
    
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
            if param is not None:
                param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    def amplitude(self, x):
        raise NotImplementedError
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)

class fMPSModel(wavefunctionModel):
    def __init__(self, ftn, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fMPS (exact contraction)':{'D': ftn.max_bond(), 'L': ftn.L, 'symmetry': self.symmetry, 'cyclic': ftn.cyclic, 'skeleton': self.skeleton},
        }

    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)
            amp_val = amp.contract()
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

class fMPS_TNFModel(wavefunctionModel):
    def __init__(self, ftn, dtype=torch.float32, max_bond=None, direction='y'):
        super().__init__()
        self.param_dtype = dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]
        self.max_bond = max_bond
        self.direction = direction
        self.depth = ftn.Lx

        self.model_structure = {
            'fMPS_TNF (1+1)D':
            {'D': ftn.max_bond(), 
             'chi': max_bond,
             'L': ftn.L, 
             'total depth': self.depth,
             'symmetry': self.symmetry, 
             'cyclic': ftn.cyclic, 
             'skeleton': self.skeleton,
             'direction': self.direction,
            },
        }

    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)
            if self.max_bond is None or self.max_bond <= 0:
                amp_val = amp.contract()
            else:
                if self.direction == 'y':
                    amp.contract_boundary_from_ymin_(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.L//2-1])
                    amp.contract_boundary_from_ymax_(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.L//2, psi.L-1])
                amp_val = amp.contract()
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

class fMPS_backflow_Model(wavefunctionModel):

    def __init__(self, ftn, nn_hidden_dim=128, nn_eta=1e-3, num_hidden_layer=1, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        # Define the neural network
        input_dim = ftn.L
        tn_params_vec = flatten_tn_params(params)
        layers = []
        current_input_dim = input_dim
        for i in range(num_hidden_layer):
            layers.append(nn.Linear(current_input_dim, nn_hidden_dim))
            # layers.append(nn.LeakyReLU())
            layers.append(nn.Tanh())
            current_input_dim = nn_hidden_dim
        layers.append(nn.Linear(current_input_dim, tn_params_vec.numel()))
        self.nn = nn.Sequential(*layers)
        self.nn.to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fMPS_backflow (exact contraction)':{'D': ftn.max_bond(), 'L': ftn.L, 'symmetry': self.symmetry},
        }

        self.nn_eta = nn_eta
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the NN correction to the parameters
            nn_correction = self.nn(x_i)
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            amp_val = amp.contract()
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

class fMPS_backflow_attn_Model(wavefunctionModel):

    def __init__(self, ftn, nn_hidden_dim=128, nn_eta=1e-3, num_attention_blocks=1, embedding_dim=16, attention_heads=4, d_inner=16, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        # Define the neural network
        input_dim = ftn.L
        tn_params_vec = flatten_tn_params(params)
        phys_dim = ftn.phys_dim()
        self.nn = StackedSelfAttn_FFNN(
            n_site=input_dim,
            num_classes=phys_dim,
            num_attention_blocks=num_attention_blocks,
            embedding_dim=embedding_dim,
            d_inner=d_inner,
            attention_heads=attention_heads,
            nn_hidden_dim=nn_hidden_dim,
            output_dim=tn_params_vec.numel(),
            dtype=self.param_dtype
        )

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fMPS_backflow_attn (exact contraction)':{'D': ftn.max_bond(), 'L': ftn.L, 'symmetry': self.symmetry, 'num_attention_blocks': num_attention_blocks, 'embedding_dim': embedding_dim, 'attention_heads': attention_heads, 'd_inner': d_inner},
        }

        self.nn_eta = nn_eta
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the NN correction to the parameters
            nn_correction = self.nn(x_i)
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            amp_val = amp.contract()
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
        

#------------ fermionic PEPS based model ------------

class fTNModel(wavefunctionModel):

    def __init__(self, ftn, max_bond=None, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            f'fPEPS (chi={max_bond})':{'D': ftn.max_bond(), 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry},
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.tree = None

    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)
            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.ReusableHyperOptimizer()
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree)

            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                amp_val = amp.contract()

            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class fTNModel_test(wavefunctionModel):
    """The oddpos in this model is ordered so that no global phase is generated during contraction."""

    def __init__(self, ftn, max_bond=None, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS (exact contraction)':{'D': ftn.max_bond(), 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry},
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=False, reverse=0)
            if self.max_bond is None:
                amp = amp
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
            amp_val = amp.contract()
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    

class fTN_backflow_Model(wavefunctionModel):

    def __init__(self, ftn, max_bond=None, nn_hidden_dim=128, nn_eta=1e-3, num_hidden_layer=1, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Define the neural network
        input_dim = ftn.Lx * ftn.Ly
        tn_params_vec = flatten_tn_params(params)
        layers = []
        current_input_dim = input_dim
        for i in range(num_hidden_layer):
            layers.append(nn.Linear(current_input_dim, nn_hidden_dim))
            layers.append(nn.LeakyReLU())
            current_input_dim = nn_hidden_dim
        layers.append(nn.Linear(current_input_dim, tn_params_vec.numel()))
        self.nn = nn.Sequential(*layers)
        self.nn.to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS _backflow':{'D': ftn.max_bond(), 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry, 'nn_hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'max_bond': max_bond},
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.nn_eta = nn_eta
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the NN correction to the parameters
            nn_correction = self.nn(x_i)
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])

            amp_val = amp.contract()
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

class fTN_backflow_Model_embedding(wavefunctionModel):

    def __init__(self, ftn, num_class=4, embedding_dim=32, max_bond=None, nn_hidden_dim=128, nn_eta=1e-3, num_hidden_layer=1, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Define the embedding layer for each on-site configuration
        self.embedding = nn.Linear(num_class, embedding_dim)

        # Define the neural network
        input_dim = ftn.Lx * ftn.Ly * embedding_dim
        tn_params_vec = flatten_tn_params(params)
        layers = []
        current_input_dim = input_dim
        for i in range(num_hidden_layer):
            layers.append(nn.Linear(current_input_dim, nn_hidden_dim))
            layers.append(nn.LeakyReLU())
            current_input_dim = nn_hidden_dim
        layers.append(nn.Linear(current_input_dim, tn_params_vec.numel()))
        self.nn = nn.Sequential(*layers)
        self.nn.to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS _backflow':{'D': ftn.max_bond(), 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry, 'nn_hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'max_bond': max_bond},
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.nn_eta = nn_eta
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)

        # `x` is expected to be batched as (batch_size, input_dim)
        # One-hot encode the input sequence
        one_hot_encoded = F.one_hot(x.long(), num_classes=self.embedding.in_features).float()

        # Apply the embedding to each on-site configuration
        x_embedded = self.embedding(one_hot_encoded)
        x_embedded_flat = x_embedded.view(x_embedded.size(0), -1)  # Flatten the embedding dimensions

        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i, x_embedded_i in zip(x, x_embedded_flat):
            # Check x_embedded_i type
            if not type(x_embedded_i) == torch.Tensor:
                x_embedded_i = torch.tensor(x_embedded_i, dtype=self.param_dtype)
            else:
                if x_embedded_i.dtype != self.param_dtype:
                    x_embedded_i = x_embedded_i.to(self.param_dtype)
            # Get the NN correction to the parameters
            nn_correction = self.nn(x_embedded_i)
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta * nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly // 2 - 1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly // 2, psi.Ly - 1])

            amp_val = amp.contract()
            if amp_val == 0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)



class fTN_backflow_Model_Blockwise(wavefunctionModel):
    def __init__(self, ftn, block_size=(2, 2), max_bond=None, nn_hidden_dim=128, block_nn_last_dim=4, nn_eta=1e-3, num_hidden_layer=1, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # Store symmetry and other properties
        self.symmetry = ftn.arrays[0].symmetry
        self.block_size = block_size
        Lx, Ly = ftn.Lx, ftn.Ly

        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        
        # Initialize bulk and block storage
        self.block_ts = {}
        self.nn_blocks = nn.ModuleDict()
        
        # Partition the lattice into blocks
        block_id = 0
        for x_start in range(0, Lx, block_size[0]):
            for y_start in range(0, Ly, block_size[1]):
                x_end = min(x_start + block_size[0], Lx)
                y_end = min(y_start + block_size[1], Ly)
                
                # Define the site labels for this block
                block_sites = [f"I{x},{y}" for x in range(x_start, x_end) for y in range(y_start, y_end)]
                
                # Partition the tensors into blocks and store the block tensors tid
                tid_list = [next(iter(ftn.tag_map[s])) for s in block_sites]
                self.block_ts[block_id] = tid_list

                # Create a neural network for this block, first flatten the parameters according to tid in the whole TN
                block_tn_params = {
                    tid: params[tid]
                    for tid in tid_list
                }
                block_tn_params_vec = flatten_tn_params(block_tn_params)
                input_dim = Lx * Ly
                layers = []
                current_input_dim = input_dim
                for l in range(num_hidden_layer):
                    if l == num_hidden_layer - 1:
                        layers.append(nn.Linear(current_input_dim, block_nn_last_dim))
                        current_input_dim = block_nn_last_dim
                    else:
                        layers.append(nn.Linear(current_input_dim, nn_hidden_dim))
                        current_input_dim = nn_hidden_dim
                    layers.append(nn.LeakyReLU())

                layers.append(nn.Linear(current_input_dim, block_tn_params_vec.numel()))
                self.nn_blocks[str(block_id)] = nn.Sequential(*layers)
                
                block_id += 1

        # Convert NNs to the appropriate data type
        self.nn_blocks.to(self.param_dtype)

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry
        
        # Store model details
        self.model_structure = {
            'fPEPS_backflow_blockwise': {'D': ftn.max_bond(), 'Lx': ftn.Lx, 'Ly': ftn.Ly, 
                                         'symmetry': self.symmetry, 'nn_hidden_dim': nn_hidden_dim, 'block_nn_last_dim': block_nn_last_dim,
                                         'nn_eta': nn_eta, 'max_bond': max_bond},
        }
        self.nn_eta = nn_eta
        self.max_bond = max_bond if max_bond and max_bond > 0 else None
    
    def amplitude(self, x):
        tn_params = {}
        
        batch_amps = []
        for x_i in x:
            if not isinstance(x_i, torch.Tensor):
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            
            # Neural network corrections for each block
            for block_id, nn in self.nn_blocks.items():
                block_tid_list = self.block_ts[int(block_id)]
                block_tn_params = {
                    tid: {
                        ast.literal_eval(sector): data
                        for sector, data in self.torch_tn_params[str(tid)].items()
                    }
                    for tid in block_tid_list
                }
                block_tn_params_vec = flatten_tn_params(block_tn_params)
                nn_correction = nn(x_i)
                block_tn_params_vec = block_tn_params_vec + self.nn_eta * nn_correction
                block_tn_params = reconstruct_proj_params(block_tn_params_vec, block_tn_params)
                tn_params.update(block_tn_params)
            
            psi = qtn.unpack(tn_params, self.skeleton)
            # Compute the amplitude for the corrected TN
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is not None:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, amp.Ly // 2 - 1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[amp.Ly // 2, amp.Ly - 1])

            amp_val = amp.contract()
            batch_amps.append(amp_val if amp_val != 0.0 else torch.tensor(0.0))

        return torch.stack(batch_amps)


#------------ fermionic TN model with attention mechanism ------------
from .nn_sublayers import *

class PureAttention_Model(wavefunctionModel):
    def __init__(self, phys_dim=4, n_site=None, num_attention_blocks=1, embedding_dim=32, attention_heads=4, nn_hidden_dim=128, nn_eta=1e-3, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype

        # Define the neural network
        input_dim = n_site
        self.nn = StackedSelfAttn_FFNN(
            n_site=input_dim,
            num_classes=phys_dim,
            num_attention_blocks=num_attention_blocks,
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            nn_hidden_dim=nn_hidden_dim,
            output_dim=1
        )

        self.model_structure = {
            'pure attention':
            {'n_site': n_site, 
             'phys_dim': phys_dim, 
             'num_attention_blocks': num_attention_blocks, 
             'embedding_dim': embedding_dim, 
             'attention_heads': attention_heads, 
             'nn_hidden_dim': nn_hidden_dim, 
             'nn_eta': nn_eta
            },
        }
        self.nn_eta = nn_eta
        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]
    
    
    def amplitude(self, x):
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i)
            amp = self.nn(x_i)
            batch_amps.append(amp.squeeze())
        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

class fTN_backflow_attn_Model(wavefunctionModel):
    def __init__(self, ftn, max_bond=None, embedding_dim=32, attention_heads=4, nn_hidden_dim=128, nn_eta=1e-3, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Define the neural network
        input_dim = ftn.Lx * ftn.Ly
        phys_dim = ftn.phys_dim()
        tn_params_vec = flatten_tn_params(params)
        
        self.nn = SelfAttn_FFNN_block(
            n_site=input_dim,
            num_classes=phys_dim,
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            nn_hidden_dim=nn_hidden_dim,
            output_dim=tn_params_vec.numel(),
            dtype=self.param_dtype
        )

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS_backflow_attn':{'D': ftn.max_bond(), 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry, 'nn_hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'max_bond': max_bond, 'embedding_dim': embedding_dim, 'attention_heads': attention_heads},
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.nn_eta = nn_eta
        self.tree = None
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the NN correction to the parameters
            nn_correction = self.nn(x_i)
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.ReusableHyperOptimizer()
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree)
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                amp_val = amp.contract()

            if amp_val==0.0:
                amp_val = torch.tensor(0.0)

            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

class fTN_backflow_attn_Jastrow_Model(wavefunctionModel):
    def __init__(self, ftn, max_bond=None, embedding_dim=32, attention_heads=4, nn_hidden_dim=128, nn_eta=1e-3, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Define the neural network
        input_dim = ftn.Lx * ftn.Ly
        phys_dim = ftn.phys_dim()
        tn_params_vec = flatten_tn_params(params)
        
        self.nn = SelfAttn_FFNN_block(
            n_site=input_dim,
            num_classes=phys_dim,
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            nn_hidden_dim=nn_hidden_dim,
            output_dim=tn_params_vec.numel()
        )

        self.Jastrow = nn.Sequential(
            nn.Linear(input_dim, nn_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(nn_hidden_dim, 1)
        )

        self.Jastrow.to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS_backflow_attn':{'D': ftn.max_bond(), 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry, 'nn_hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'max_bond': max_bond, 'embedding_dim': embedding_dim, 'attention_heads': attention_heads},
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.nn_eta = nn_eta
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Get the NN correction to the parameters
            nn_correction = self.nn(x_i)
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])

            amp_val = amp.contract()
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            jastrow_factor = sum(self.Jastrow(x_i))
            batch_amps.append(amp_val*torch.exp(jastrow_factor))

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class fTN_backflow_attn_Model_Stacked(wavefunctionModel):
    def __init__(
            self, 
            ftn, 
            max_bond=None, 
            num_attention_blocks=1, 
            embedding_dim=16, 
            d_inner=16, 
            attention_heads=4, 
            nn_hidden_dim=128, 
            nn_eta=1e-3, 
            dtype=torch.float32
        ):
        super().__init__()
        self.param_dtype = dtype

        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Define the neural network
        input_dim = ftn.Lx * ftn.Ly
        phys_dim = ftn.phys_dim()
        tn_params_vec = flatten_tn_params(params)
        
        self.nn = StackedSelfAttn_FFNN(
            n_site=input_dim,
            num_classes=phys_dim,
            num_attention_blocks=num_attention_blocks,
            embedding_dim=embedding_dim,
            d_inner=d_inner,
            attention_heads=attention_heads,
            nn_hidden_dim=nn_hidden_dim,
            output_dim=tn_params_vec.numel(),
        )

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS_backflow_attn_stacked':{
                'D': ftn.max_bond(),
                'max_bond': max_bond, 
                'Lx': ftn.Lx, 
                'Ly': ftn.Ly, 
                'symmetry': self.symmetry, 
                'num_attention_blocks': num_attention_blocks,
                'nn_hidden_dim': nn_hidden_dim,
                'embedding_dim': embedding_dim,
                'd_inner': d_inner,
                'nn_eta': nn_eta, 
            },
        }

        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.nn_eta = nn_eta
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i)
            # Get the NN correction to the parameters
            nn_correction = self.nn(x_i)
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])

            amp_val = amp.contract()
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class fTN_backflow_attn_Model_boundary(wavefunctionModel):
    def __init__(self, ftn, max_bond=None, embedding_dim=32, attention_heads=4, nn_hidden_dim=128, nn_eta=1e-3, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get the boundary tensors tids and parameter shapes
        boundary_sites = [site for site in ftn.gen_site_coos() if any(c == 0 or c == ftn.Lx - 1 for c in site)]
        boundary_tags = [ftn.site_tag_id.format(*site) for site in boundary_sites]
        bulk_site_tags = [ftn.site_tag_id.format(*site) for site in ftn.gen_site_coos() if site not in boundary_sites]
        self.bulk_tid_list = [next(iter(ftn._get_tids_from_tags([tag]))) for tag in bulk_site_tags]
        self.boundary_tid_list = [next(iter(ftn._get_tids_from_tags([tag]))) for tag in boundary_tags]
        boundary_tn_params = {
            tid: params[tid]
            for tid in self.boundary_tid_list
        }
        boundary_tn_params_vec = flatten_tn_params(boundary_tn_params)

        # Define the neural network for the backflow transformation to boundary tensors
        input_dim = ftn.Lx * ftn.Ly
        phys_dim = ftn.phys_dim()
        
        self.nn = SelfAttn_FFNN_block(
            n_site=input_dim,
            num_classes=phys_dim,
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            nn_hidden_dim=nn_hidden_dim,
            output_dim=boundary_tn_params_vec.numel(),
            dtype=self.param_dtype
        )

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS_backflow_attn_boundary':
            {
                'D': ftn.max_bond(), 
                'Lx': ftn.Lx, 'Ly': ftn.Ly, 
                'symmetry': self.symmetry, 
                'nn_hidden_dim': nn_hidden_dim, 
                'nn_eta': nn_eta, 
                'embedding_dim': embedding_dim,
                'attention_heads': attention_heads,
                'max_bond': max_bond,
            },
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.nn_eta = nn_eta
        self.tree = None
    
    def amplitude(self, x):
        tn_nn_params = {}

        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            
            # Get the bulk parameters
            bulk_tn_params = {
                tid: {
                    ast.literal_eval(sector): data
                    for sector, data in self.torch_tn_params[str(tid)].items()
                }
                for tid in self.bulk_tid_list
            }
            tn_nn_params.update(bulk_tn_params)
            
            # Get the boundary parameters
            boundary_tn_params = {
                tid: {
                    ast.literal_eval(sector): data
                    for sector, data in self.torch_tn_params[str(tid)].items()
                }
                for tid in self.boundary_tid_list
            }
            boundary_tn_params_vec = flatten_tn_params(boundary_tn_params)

            # Get the NN correction to the boundary parameters
            nn_correction = self.nn(x_i)
            # Add the correction to the original parameters
            new_boundary_tn_params_vec = boundary_tn_params_vec + self.nn_eta*nn_correction
            new_boundary_tn_params = reconstruct_proj_params(new_boundary_tn_params_vec, boundary_tn_params)
            tn_nn_params.update(new_boundary_tn_params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.ReusableHyperOptimizer()
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree)
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                amp_val = amp.contract()
                
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class fTN_backflow_attn_Tensorwise_Model(wavefunctionModel):
    """
        For each on-site fermionic tensor with specific shape, assign a narrow on-site projector MLP with corresponding output dimension.
        This is to avoid the large number of parameters in the previous model, where Np = N_neurons * N_TNS.
    """
    ...
    def __init__(self, ftn, max_bond=None, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # for each tensor (labelled by tid), assign a attention+MLP
        self.nn = nn.ModuleDict()
        for tid in self.torch_tn_params.keys():
            input_dim = ftn.Lx * ftn.Ly
            phys_dim = ftn.phys_dim()
            tn_params_dict ={
                tid: params[int(tid)]
            }
            tn_params_vec = flatten_tn_params(tn_params_dict)
            self.nn[tid] = SelfAttn_FFNN_block(
                n_site=input_dim,
                num_classes=phys_dim,
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                nn_hidden_dim=nn_final_dim,
                output_dim=tn_params_vec.numel(),
                dtype=self.param_dtype
            )

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS_backflow_attn_Tensorwise':
            {
                'D': ftn.max_bond(), 
                'Lx': ftn.Lx, 'Ly': ftn.Ly, 
                'symmetry': self.symmetry, 
                'nn_final_dim': nn_final_dim,
                'nn_eta': nn_eta, 
                'embedding_dim': embedding_dim,
                'attention_heads': attention_heads,
                'max_bond': max_bond,
            },
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.nn_eta = nn_eta
        self.tree = None
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)

        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
        
            # Get the NN correction to the parameters, concatenate the results for each tensor
            nn_correction = torch.cat([self.nn[tid](x_i) for tid in self.torch_tn_params.keys()])
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.ReusableHyperOptimizer()
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree)
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                amp_val = amp.contract()
                
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    

class fTN_backflow_attn_Tensorwise_Model_v1(wavefunctionModel):
    """
        For each on-site fermionic tensor with specific shape, assign a narrow on-site projector MLP with corresponding output dimension.
        This is to avoid the large number of parameters in the previous model, where Np = N_neurons * N_TNS.
    """
    ...
    def __init__(self, ftn, max_bond=None, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Define the neural network
        input_dim = ftn.Lx * ftn.Ly
        phys_dim = ftn.phys_dim()
        
        self.nn = SelfAttn_block(
            n_site=input_dim,
            num_classes=phys_dim,
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            dtype=self.param_dtype
        )
        # for each tensor (labelled by tid), assign a MLP
        self.mlp = nn.ModuleDict()
        for tid in self.torch_tn_params.keys():
            mlp_input_dim = ftn.Lx * ftn.Ly * embedding_dim
            tn_params_dict = {
                tid: params[int(tid)]
            }
            tn_params_vec = flatten_tn_params(tn_params_dict)
            self.mlp[tid] = nn.Sequential(
                nn.Linear(mlp_input_dim, nn_final_dim),
                nn.ReLU(),
                nn.Linear(nn_final_dim, tn_params_vec.numel()),
            )
            self.mlp[tid].to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS_backflow_attn_Tensorwise':
            {
                'D': ftn.max_bond(), 
                'Lx': ftn.Lx, 'Ly': ftn.Ly, 
                'symmetry': self.symmetry, 
                # 'nn_hidden_dim': nn_hidden_dim, 
                'nn_final_dim': nn_final_dim,
                'nn_eta': nn_eta, 
                'embedding_dim': embedding_dim,
                'attention_heads': attention_heads,
                'max_bond': max_bond,
            },
        }
        if max_bond is None or max_bond <= 0:
            max_bond = None
        self.max_bond = max_bond
        self.nn_eta = nn_eta
        self.tree = None
    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)

        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
        
            # Get the NN correction to the parameters, concatenate the results for each tensor
            nn_features = self.nn(x_i)
            nn_features_vec = nn_features.view(-1)
            nn_correction = torch.cat([self.mlp[tid](nn_features_vec) for tid in self.torch_tn_params.keys()])
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.ReusableHyperOptimizer()
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree)
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                amp_val = amp.contract()
                
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

#------------ fermionic TN model with NN projectors insertion ------------

    
class fTN_NN_proj_Model(torch.nn.Module):
    
    def __init__(self, ftn, max_bond, nn_hidden_dim=64, nn_eta=1e-3, param_dtype=torch.float32):
        super().__init__()
        self.max_bond = max_bond if max_bond > 0 else None
        self.nn_eta = nn_eta
        self.param_dtype = param_dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        
        self.parity_config = [array.parity for array in ftn.arrays]
        self.N_fermion = sum(self.parity_config)
        dummy_config = torch.zeros(ftn.nsites)
        dummy_config[:self.N_fermion] = 1
        dummy_amp = self.get_amp(ftn, dummy_config, inplace=False)
        dummy_amp_w_proj = insert_proj_peps(dummy_amp, max_bond=max_bond, yrange=[0, ftn.Ly-2])
        dummy_amp_tn, dummy_proj_tn = dummy_amp_w_proj.partition(tags='proj')
        dummy_proj_params, dummy_proj_skeleton = qtn.pack(dummy_proj_tn)
        dummy_proj_params_vec = flatten_tn_params(dummy_proj_params)
        self.proj_params_vec_len = len(dummy_proj_params_vec)

        # Define an MLP layer (or any other neural network layers)
        self.mlp = nn.Sequential(
            nn.Linear(ftn.nsites, nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(nn_hidden_dim, self.proj_params_vec_len)
        )

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry
        assert self.symmetry == 'Z2', "Only Z2 symmetry fPEPS is supported for NN insertion now."
        if self.symmetry == 'Z2':
            assert self.N_fermion %2 == sum(self.parity_config) % 2, "The number of fermions must match the parity of the Z2-TNS."

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS (proj inserted)':{'D': ftn.max_bond(), 'chi': self.max_bond, 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry, 'proj_yrange': [0, ftn.Ly-2]},
            '2LayerMLP':{'hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'activation': 'ReLU'}
        }
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    @property
    def num_tn_params(self):
        num=0
        for tid, blk_array in self.torch_tn_params.items():
            for sector, data in blk_array.items():
                num += data.numel()
        return num
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param

    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            amp = psi.get_amp(psi, x_i)

            # Insert projectors
            amp_w_proj = insert_proj_peps(amp, max_bond=self.max_bond, yrange=[0, psi.Ly-2])
            amp_tn, proj_tn = amp_w_proj.partition(tags='proj')
            proj_params, proj_skeleton = qtn.pack(proj_tn)
            proj_params_vec = flatten_tn_params(proj_params)

            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            # Add NN output
            proj_params_vec += self.nn_eta*self.mlp(x_i)
            # Reconstruct the proj parameters
            new_proj_params = reconstruct_proj_params(proj_params_vec, proj_params)
            # Load the new parameters
            new_proj_tn = qtn.unpack(new_proj_params, proj_skeleton)
            new_amp_w_proj = amp_tn | new_proj_tn

            # batch_amps.append(torch.tensor(new_amp_w_proj.contract(), dtype=torch.float32, requires_grad=True))
            batch_amps.append(new_amp_w_proj.contract())

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)


class fTN_NN_proj_variable_Model(torch.nn.Module):
    
    def __init__(self, ftn, max_bond, nn_hidden_dim=64, nn_eta=1e-3, dtype=torch.float32, padded_length=0, dummy_config=None, lazy=False):
        super().__init__()
        self.max_bond = max_bond if max_bond > 0 else None
        self.nn_eta = nn_eta
        self.param_dtype = dtype
        self.padded_length = padded_length
        self.lazy = lazy
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        
        self.parity_config = [array.parity for array in ftn.arrays]
        self.N_fermion = sum(self.parity_config)
        assert dummy_config is not None, "Please provide a dummy configuration for the model."
        dummy_amp = ftn.get_amp(dummy_config)
        dummy_amp_w_proj = insert_proj_peps(dummy_amp, max_bond=max_bond, yrange=[0, ftn.Ly-2], lazy=lazy)
        dummy_amp_tn, dummy_proj_tn = dummy_amp_w_proj.partition(tags='proj')
        dummy_proj_params, dummy_proj_skeleton = qtn.pack(dummy_proj_tn)
        dummy_proj_params_vec = flatten_tn_params(dummy_proj_params)
        self.proj_params_vec_len = len(dummy_proj_params_vec) + padded_length

        # Define an MLP layer (or any other neural network layers)
        self.mlp = nn.Sequential(
            nn.Linear(ftn.nsites, nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(nn_hidden_dim, self.proj_params_vec_len)
        )
        self.mlp.to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry
        if self.symmetry == 'Z2':
            assert self.N_fermion %2 == sum(self.parity_config) % 2, "The number of fermions must match the parity of the Z2-TNS."

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS (variable length proj)':{'D': ftn.max_bond(), 'chi': self.max_bond, 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry, 'proj_yrange': [0, ftn.Ly-2], 'padded_length': padded_length},
            '2LayerMLP':{'hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'activation': 'ReLU'}
        }
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    @property
    def num_tn_params(self):
        num=0
        for tid, blk_array in self.torch_tn_params.items():
            for sector, data in blk_array.items():
                num += data.numel()
        return num
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param

    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            amp = psi.get_amp(x_i)

            # Insert projectors
            try:
                amp_w_proj = insert_proj_peps(amp, max_bond=self.max_bond, yrange=[0, psi.Ly-2], lazy=self.lazy)
            except:
                print('ill configuration:', x_i)
                amp_val = torch.tensor(0.0, dtype=self.param_dtype)
                batch_amps.append(amp_val)
                continue

            amp_tn, proj_tn = amp_w_proj.partition(tags='proj')
            proj_params, proj_skeleton = qtn.pack(proj_tn)
            proj_params_vec = flatten_tn_params(proj_params)

            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Add NN output
            try:
                proj_params_vec += self.nn_eta*self.mlp(x_i)[:len(proj_params_vec)]
            except:
                raise ValueError(f'Shape mismatch error, proj_params_vec: {len(proj_params_vec)}, mlp(x_i): {self.mlp(x_i).shape}')
            # Reconstruct the proj parameters
            new_proj_params = reconstruct_proj_params(proj_params_vec, proj_params)
            # Load the new parameters
            new_proj_tn = qtn.unpack(new_proj_params, proj_skeleton)
            new_amp_w_proj = amp_tn | new_proj_tn

            amp_val = new_amp_w_proj.contract()
            if amp_val == 0:
                amp_val = torch.tensor(0.0, dtype=self.param_dtype)
            # batch_amps.append(torch.tensor(new_amp_w_proj.contract(), dtype=torch.float32, requires_grad=True))
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)


class fTN_NN_2row_Model(torch.nn.Module):
    """2-row fTN with NN insertion."""
    def __init__(self, ftn, max_bond, nn_hidden_dim=64, nn_eta=1e-3, param_dtype=torch.float32):
        super().__init__()
        self.max_bond = max_bond if max_bond > 0 else None
        self.nn_eta = nn_eta
        self.param_dtype = param_dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        
        self.charge_config = [array.charge for array in ftn.arrays]
        self.N_fermion = sum(self.charge_config)
        dummy_config = torch.zeros(ftn.nsites)
        dummy_config[:self.N_fermion] = 1
        dummy_amp = ftn.get_amp(dummy_config, inplace=False, conj=True)
        dummy_amp_2row = dummy_amp.contract_boundary_from_ymin(max_bond=max_bond, cutoff=0.0, yrange=[0, ftn.Ly//2-1])
        dummy_amp_2row = dummy_amp_2row.contract_boundary_from_ymax(max_bond=max_bond, cutoff=0.0, yrange=[ftn.Ly//2, ftn.Ly-1])
        dummy_2row_params, dummy_2row_skeleton = qtn.pack(dummy_amp_2row)
        dummy_2row_params_vec = flatten_tn_params(dummy_2row_params)
        self.tworow_params_vec_len = len(dummy_2row_params_vec)

        # Define an MLP layer (or any other neural network layers)
        self.mlp = nn.Sequential(
            nn.Linear(ftn.nsites, nn_hidden_dim),
            nn.ReLU(),
            nn.Linear(nn_hidden_dim, self.tworow_params_vec_len)
        )

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry
        assert self.symmetry == 'Z2', "Only Z2 symmetry fPEPS is supported for NN insertion now."
        if self.symmetry == 'Z2':
            assert self.N_fermion %2 == sum(self.charge_config) % 2, "The number of fermions must match the parity of the Z2-TNS."

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS (2-row)':{'D': ftn.max_bond(), 'chi': self.max_bond, 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry},
            '2LayerMLP':{'hidden_dim': nn_hidden_dim, 'nn_eta': nn_eta, 'activation': 'ReLU'}
        }
        
    
    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    @property
    def num_tn_params(self):
        num=0
        for tid, blk_array in self.torch_tn_params.items():
            for sector, data in blk_array.items():
                num += data.numel()
        return num
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param

    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            amp = psi.get_amp(x_i, conj=True)
            # Contract to 2 rows
            amp_2row = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
            amp_2row = amp_2row.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
            amp_2row_params, amp_2row_skeleton = qtn.pack(amp_2row)
            amp_2row_params_vec = flatten_tn_params(amp_2row_params)
            
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            # Add NN output
            amp_2row_params_vec = amp_2row_params_vec + self.nn_eta*self.mlp(x_i)
            # Reconstruct the proj parameters
            new_2row_params = reconstruct_proj_params(amp_2row_params_vec, amp_2row_params)
            # Load the new parameters
            new_amp_2row = qtn.unpack(new_2row_params, amp_2row_skeleton)

            batch_amps.append(new_amp_2row.contract())

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)
    

# ----------------- Transformer model -----------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dtype=torch.float32):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(
            self, 
            output_size=1,
            phys_dim=2,
            d_model=128, 
            nhead=8, 
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dim_feedforward=512, 
            dropout=0.1,
            dtype=torch.float32
        ):
        super(TransformerModel, self).__init__()
        self.dtype = dtype
        # Embedding layer for integer input sequence
        self.embedding = nn.Embedding(phys_dim, d_model)
        # Embedding layer for floating-point input sequence
        self.float_embedding = nn.Linear(1, d_model, dtype=dtype)
        # Positional encoding for fixed-length sequences
        self.pos_encoder = PositionalEncoding(d_model, dtype=dtype)
        # Transformer layers
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, dtype=dtype, batch_first=True)
        # Linear layer for output generation
        self.fc_out = nn.Linear(d_model, output_size, dtype=dtype)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source (input) sequence
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=self.dtype)) # [batch_size, seq_len, d_model]
        src = self.pos_encoder(src)
        # Encode target (output) sequence
        tgt = self.float_embedding(tgt) * torch.sqrt(torch.tensor(self.float_embedding.out_features, dtype=self.dtype))
        tgt = self.pos_encoder(tgt)
        # Apply transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        # Generate output for each position in the sequence
        output = self.fc_out(output) # [batch_size, seq_len, output_size]
        return output



class fTN_Transformer_Model(torch.nn.Module):
    max_bond: int
    nn_eta: float
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    dtype: torch.dtype

    def __init__(
            self, 
            ftn, 
            max_bond, 
            nn_eta=1e-3, 
            d_model=128, 
            nhead=8, 
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dim_feedforward=512, 
            dropout=0.0,
            dtype=torch.float32
        ):

        super().__init__()
        self.max_bond = max_bond if max_bond > 0 else None
        self.nn_eta = nn_eta
        self.phys_dim = ftn.phys_dim()
        self.param_dtype = dtype
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)
        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        self.parity_config = [array.parity for array in ftn.arrays]
        self.N_fermion = sum(self.parity_config)
        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry
        assert self.symmetry == 'Z2' or self.symmetry == 'U1', "Only Z2 or U1 symmetry fPEPS is supported for Transformer insertion now."
        if self.symmetry == 'Z2':
            assert self.N_fermion %2 == sum(self.parity_config) % 2, "The number of fermions must match the parity of the Z2-TNS."
        
        # Transformer model
        self.d_model = d_model # Embedding dimension
        self.transformer = TransformerModel(
            output_size=1, 
            phys_dim=self.phys_dim,
            d_model=self.d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            dtype=self.param_dtype,
        )

        # Store the shapes of the parameters XXX: needed at all?
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS (transformer-two-row)':{'D': ftn.max_bond(), 'chi': self.max_bond, 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry},
            'transformer':{'input_size': ftn.nsites, 'output_size': 1}
        }

    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    @property
    def num_tn_params(self):
        num=0
        for tid, blk_array in self.torch_tn_params.items():
            for sector, data in blk_array.items():
                num += data.numel()
        return num
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param

    def amplitude(self, x):
        assert x.ndim != 1, "Amplitude input must be batched."
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            amp = psi.get_amp(x_i, conj=True)
            # Contract to 2 rows
            amp_2row = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
            amp_2row = amp_2row.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
            amp_2row_params, amp_2row_skeleton = qtn.pack(amp_2row)
            amp_2row_params_vec = flatten_tn_params(amp_2row_params)
            
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=torch.long)
            else:
                x_i = x_i.to(torch.long)

            # Input of the transformer
            src = x_i.unsqueeze(0) # Shape: [batch_size==1, seq_len]
            # Target of the transformer
            tgt = torch.tensor(amp_2row_params_vec, dtype=self.param_dtype).unsqueeze(0) # Shape: [batch_size==1, seq_len]
            tgt.unsqueeze_(-1) # Shape: [batch_size==1, seq_len, 1]
            # Forward pass
            nn_output = self.transformer(src, tgt)
            # concatenate the output to get the final vector of length vec_len
            nn_output = nn_output.view(-1)
            # Add NN output
            amp_2row_params_vec = amp_2row_params_vec + self.nn_eta*nn_output
            # Reconstruct the 2-row TN parameters
            new_2row_params = reconstruct_proj_params(amp_2row_params_vec, amp_2row_params)
            # Load the new parameters
            new_amp_2row = qtn.unpack(new_2row_params, amp_2row_skeleton)
            # Add to batch
            batch_amps.append(new_amp_2row.contract())

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)

class fTN_Transformer_Proj_lazy_Model(torch.nn.Module):
    def __init__(
            self, 
            ftn, 
            max_bond, 
            nn_eta=1e-3, 
            d_model=128, 
            nhead=8, 
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dim_feedforward=512, 
            dropout=0.1,
            dtype=torch.float32,
            lazy=True,
        ):
        super().__init__()
        self.max_bond = max_bond if max_bond > 0 else None
        self.nn_eta = nn_eta
        self.phys_dim = ftn.phys_dim()
        self.param_dtype = dtype
        self.lazy = lazy
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        
        self.charge_config = [array.charge for array in ftn.arrays]
        self.N_fermion = sum(self.charge_config)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry
        assert self.symmetry == 'Z2' or self.symmetry == 'U1', "Only Z2 or U1 symmetry fPEPS is supported for Transformer insertion now."
        if self.symmetry == 'Z2':
            assert self.N_fermion %2 == sum(self.charge_config) % 2, "The number of fermions must match the parity of the Z2-TNS."
        
        # Transformer model
        self.d_model = d_model # embedding dimension
        self.transformer = TransformerModel(
            output_size=1,
            phys_dim=self.phys_dim,
            d_model=self.d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            dtype=self.param_dtype,
        )

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS (transformer-proj)':{'D': ftn.max_bond(), 'chi': self.max_bond, 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry},
            'transformer':{'input_size': ftn.nsites, 'output_size': 1}
        }

    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    @property
    def num_tn_params(self):
        num=0
        for tid, blk_array in self.torch_tn_params.items():
            for sector, data in blk_array.items():
                num += data.numel()
        return num
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param

    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            amp = psi.get_amp(x_i, conj=True)
            # Insert projectors
            try:
                amp_w_proj = insert_proj_peps(amp, max_bond=self.max_bond, yrange=[0, psi.Ly-2], lazy=self.lazy)
            except:
                print('ill configuration:', x_i)
                amp_value = torch.tensor(0.0, dtype=self.param_dtype)
                batch_amps.append(amp_value)
                continue
            
            amp_tn, proj_tn = amp_w_proj.partition(tags='proj')
            proj_params, proj_skeleton = qtn.pack(proj_tn)
            proj_params_vec = flatten_tn_params(proj_params)
            
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=torch.int64)
            else:
                x_i = x_i.to(torch.int64)

            # Input of the transformer
            src = x_i.unsqueeze(0) # Shape: [batch_size==1, seq_len]
            # Target of the transformer
            tgt = torch.tensor(proj_params_vec, dtype=self.param_dtype).unsqueeze(0) # Shape: [batch_size==1, seq_len]
            tgt.unsqueeze_(-1) # Shape: [batch_size==1, seq_len, 1]
            # Forward pass
            nn_output = self.transformer(src, tgt)
            # concatenate the output to get the final vector of length vec_len
            nn_output = nn_output.view(-1)
            # Add NN output
            proj_params_vec = proj_params_vec + self.nn_eta*nn_output
            # Reconstruct the proj parameters
            new_proj_params = reconstruct_proj_params(proj_params_vec, proj_params)
            # Load the new parameters
            new_proj_tn = qtn.unpack(new_proj_params, proj_skeleton)
            # Add to batch
            new_amp_w_proj = amp_tn | new_proj_tn
            amp_value = new_amp_w_proj.contract()
            if amp_value == 0:
                amp_value = torch.tensor(0.0, dtype=self.param_dtype)
            batch_amps.append(amp_value)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)

class fTN_Transformer_Proj_Model(torch.nn.Module):
    def __init__(
            self, 
            ftn, 
            max_bond, 
            nn_eta=1e-3, 
            d_model=128, 
            nhead=8, 
            num_encoder_layers=6, 
            num_decoder_layers=6, 
            dim_feedforward=512, 
            dropout=0.1,
            dtype=torch.float32,
            lazy=False,
        ):
        super().__init__()
        self.max_bond = max_bond if max_bond > 0 else None
        self.nn_eta = nn_eta
        self.phys_dim = ftn.phys_dim()
        self.param_dtype = dtype
        self.lazy = lazy
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        
        self.charge_config = [array.charge for array in ftn.arrays]
        self.N_fermion = sum(self.charge_config)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry
        assert self.symmetry == 'Z2' or self.symmetry == 'U1', "Only Z2 or U1 symmetry fPEPS is supported for Transformer insertion now."
        if self.symmetry == 'Z2':
            assert self.N_fermion %2 == sum(self.charge_config) % 2, "The number of fermions must match the parity of the Z2-TNS."
        
        # Transformer model
        self.d_model = d_model # embedding dimension
        self.transformer = TransformerModel(
            output_size=1,
            phys_dim=self.phys_dim,
            d_model=self.d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            dtype=self.param_dtype,
        )

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS (transformer-two-row)':{'D': ftn.max_bond(), 'chi': self.max_bond, 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry},
            'transformer':{'input_size': ftn.nsites, 'output_size': 1}
        }

    def from_params_to_vec(self):
        return torch.cat([param.data.flatten() for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    @property
    def num_tn_params(self):
        num=0
        for tid, blk_array in self.torch_tn_params.items():
            for sector, data in blk_array.items():
                num += data.numel()
        return num
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def load_params(self, new_params):
        pointer = 0
        for param, shape in zip(self.parameters(), self.param_shapes):
            num_param = param.numel()
            new_param_values = new_params[pointer:pointer+num_param].view(shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    def reload_tn(self, old_tn, new_value_tn):
        """Reload the data of the tensors in old_tn with the data of the tensors with the same tags in new_value_tn."""
        for old_ts in old_tn:
            tags = old_ts.tags
            new_ts_ = new_value_tn.select_tensors(tags)
            assert len(new_ts_) == 1, 'Tensor must be unique.'
            new_ts = new_ts_[0]
            # print(old_ts.inds, new_ts.inds, old_ts.data.shape, new_ts.data.shape)
            old_ts.modify(data=new_ts.data)
        return old_tn
    
    def add_transformer_values(self, proj_tn, x_i):
        """Obtain the new amplitude with projectors TN by adding the output of the transformer to the projectors."""
        proj_params, proj_skeleton = qtn.pack(proj_tn)
        proj_params_vec = flatten_tn_params(proj_params)

        # Check x_i type
        if not type(x_i) == torch.Tensor:
            x_i = torch.tensor(x_i, dtype=self.param_dtype)
        else:
            if x_i.dtype != self.param_dtype:
                x_i = x_i.to(self.param_dtype)
        # Input of the transformer
        src = x_i.unsqueeze(0) # Shape: [batch_size==1, seq_len]
        # Target of the transformer
        tgt = proj_params_vec.unsqueeze(0) # Shape: [batch_size==1, seq_len]
        tgt.unsqueeze_(-1) # Shape: [batch_size==1, seq_len, 1]
        # Forward pass
        nn_output = self.transformer(src, tgt)
        # concatenate the output to get the final vector of length vec_len
        nn_output = nn_output.view(-1)
        # Add NN output
        proj_params_vec = proj_params_vec + self.nn_eta*nn_output
        # Reconstruct the proj parameters
        new_proj_params = reconstruct_proj_params(proj_params_vec, proj_params)
        # Load the new parameters
        new_proj_tn = qtn.unpack(new_proj_params, proj_skeleton)

        return new_proj_tn

    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            amp = psi.get_amp(x_i, conj=True)
            # Insert projectors
            """Insert projectors in a PEPS along the x direction towards y direction."""
            r = Rotator2D(amp, xrange=None, yrange=[0, psi.Ly-2], from_which='ymin')
            tn_calc = amp.copy()
            # tn_body_nn_value = None
            try:
                for i, inext in pairwise(r.sweep):
                    i_passed = [x for x in range(i)]
                    for j in r.sweep_other:
                        # this handles cyclic boundary conditions
                        jnext = r.get_jnext(j)
                        if jnext is not None:
                            ltags = tuple([r.site_tag(ip, j) for ip in i_passed])+(r.site_tag(i, j), r.site_tag(inext, j))
                            rtags = tuple([r.site_tag(ip, jnext) for ip in i_passed])+(r.site_tag(i, jnext), r.site_tag(inext, jnext))
                            new_ltags = (r.site_tag(inext, j),)
                            new_rtags = (r.site_tag(inext, jnext),)
                            #      │         │
                            #    ──O─┐ chi ┌─O──  j+1
                            #      │ └─▷═◁─┘│
                            #      │ ┌┘   └┐ │
                            #    ──O─┘     └─O──  j
                            #     i+1        i
                            
                            tn_calc = insert_compressor(
                                tn_calc,
                                ltags,
                                rtags,
                                new_ltags=new_ltags,
                                new_rtags=new_rtags,
                                max_bond=self.max_bond,
                            )
                
                    tn_body, proj_tn = tn_calc.partition('proj') # projectors computed from untouched TN
                    # _, tn_body_skeleton = qtn.pack(tn_body)
                    new_proj_tn = self.add_transformer_values(proj_tn, x_i) # Add transformer values to the projectors
                    
                    # if tn_body_nn_value is None:
                    #     tn_body_nn = tn_body.copy()
                    # else:
                    #     tn_body_nn = self.reload_tn(tn_body_skeleton, tn_body_nn_value)
                    
                    tn_calc = tn_body | new_proj_tn
                    
                    # contract each pair of boundary tensors with their projectors
                    for j in r.sweep_other:
                        tn_calc.contract_tags_(
                            (r.site_tag(i, j), r.site_tag(inext, j)),
                        )
                    #     tn_calc_nn.contract_tags_(
                    #         (r.site_tag(i, j), r.site_tag(inext, j)),
                    #     )
                    # tn_body_nn_value = tn_calc_nn.copy()
        
            except ZeroDivisionError:
                amp_value = torch.tensor(0.0, dtype=self.param_dtype)
                batch_amps.append(amp_value)
                continue

            # amp_value = tn_calc_nn.contract()
            amp_value = tn_calc.contract()
            if amp_value == 0:
                amp_value = torch.tensor(0.0, dtype=self.param_dtype)
            batch_amps.append(amp_value)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)


# ----------------- Neural Network Quantum States Benchmark -----------------


class SlaterDeterminant(wavefunctionModel):
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

class NeuralJastrow(wavefunctionModel):
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

class NeuralBackflow(wavefunctionModel):
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

from vmc_torch.fermion_utils import from_quimb_config_to_netket_config

class NeuralBackflow_spinful(wavefunctionModel):
    """Assuming total Sz=0."""
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64):
        super(NeuralBackflow_spinful, self).__init__()
        
        self.hilbert = hilbert
        self.param_dtype = param_dtype
        
        # Initialize the parameter M (N x Nf matrix)
        self.M = nn.Parameter(
            kernel_init(torch.empty(self.hilbert.size, self.hilbert.n_fermions_per_spin[0], dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.size, self.hilbert.n_fermions_per_spin[0], dtype=self.param_dtype)
        )

        # Initialize the neural network layer, input is n and output a matrix with the same shape as M
        self.nn = nn.Sequential(
            nn.Linear(self.hilbert.size, hidden_dim, dtype=self.param_dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.hilbert.size*self.hilbert.n_fermions_per_spin[0], dtype=self.param_dtype)
        )

        # Convert NNs to the appropriate data type
        self.nn.to(self.param_dtype)

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'Neuralbackflow':{'N_site': self.hilbert.size, 'N_fermions': self.hilbert.n_fermions, 'N_fermions_per_spin': self.hilbert.n_fermions_per_spin}
        }

    def amplitude(self, x):
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        # Define the slater determinant function manually to loop over inputs
        def backflow_det(n):
            # Compute the backflow matrix F using the neural network
            F = self.nn(n)
            M  = self.M + F.reshape(self.M.shape)
            # Find the positions of the occupied orbitals
            R = torch.nonzero(n, as_tuple=False).squeeze()

            # Extract the 2Nf x Nf submatrix of M corresponding to the occupied orbitals
            A = M[R]

            det1 = torch.linalg.det(A[:self.hilbert.n_fermions_per_spin[0]])
            det2 = torch.linalg.det(A[self.hilbert.n_fermions_per_spin[0]:])

            amp = det1*det2
            return amp
        
        batch_amps = []
        for x_i in x:
            n_i = from_quimb_config_to_netket_config(x_i)
            # Check x_i type
            if not type(n_i) == torch.Tensor:
                n_i = torch.tensor(n_i, dtype=self.param_dtype)

            amp_val=backflow_det(n_i)
            batch_amps.append(amp_val)
        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

class NNBF(wavefunctionModel):
    """Assuming total Sz=0."""
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64):
        super(NNBF, self).__init__()
        
        self.hilbert = hilbert
        self.param_dtype = param_dtype
        
        # Initialize the parameter M (N x Nf matrix)
        self.M = nn.Parameter(
            kernel_init(torch.empty(self.hilbert.size, self.hilbert.n_fermions, dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.size, self.hilbert.n_fermions, dtype=self.param_dtype)
        )

        # Initialize the neural network layer, input is n and output a matrix with the same shape as M
        self.nn = nn.Sequential(
            nn.Linear(self.hilbert.size, hidden_dim, dtype=self.param_dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.hilbert.size*self.hilbert.n_fermions, dtype=self.param_dtype)
        )

        # Convert NNs to the appropriate data type
        self.nn.to(self.param_dtype)

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'Neuralbackflow':{'N_site': self.hilbert.size, 'N_fermions': self.hilbert.n_fermions, 'N_fermions_per_spin': self.hilbert.n_fermions_per_spin}
        }

    def amplitude(self, x):
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        # Define the slater determinant function manually to loop over inputs
        def backflow_det(n):
            # Compute the backflow matrix F using the neural network
            F = self.nn(n)
            M  = self.M + F.reshape(self.M.shape)
            # Find the positions of the occupied orbitals
            R = torch.nonzero(n, as_tuple=False).squeeze()

            # Extract the 2Nf x Nf submatrix of M corresponding to the occupied orbitals
            A = M[R]
            det = torch.linalg.det(A)
            amp = det

            return amp
        
        batch_amps = []
        for x_i in x:
            n_i = from_quimb_config_to_netket_config(x_i)
            # Check x_i type
            if not type(n_i) == torch.Tensor:
                n_i = torch.tensor(n_i, dtype=self.param_dtype)

            amp_val=backflow_det(n_i)
            batch_amps.append(amp_val)
        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

class HFDS(wavefunctionModel):
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64, num_hidden_fermions=4, jastrow=False):
        super(HFDS, self).__init__()
        
        self.hilbert = hilbert
        self.Nf = self.hilbert.n_fermions
        self.Nh = num_hidden_fermions
        assert self.Nh % 2 == 0, "The number of hidden fermions must be even."
        self.jastrow = jastrow
        self.param_dtype = param_dtype
        
        # Initialize the parameter M ( Nx(Nf+Nh) matrix )
        self.M = nn.Parameter(
            kernel_init(torch.empty(self.hilbert.size, self.Nf+self.Nh, dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.size, self.Nf+self.Nh, dtype=self.param_dtype)
        )

        # Initialize Nh neural networks, input is n and output is a row vector of length Nf+Nh
        # Assume the first Nh/2 fermions are spin up and the rest are spin down
        for i in range(self.Nh):
            setattr(self, f'nn{i}', nn.Sequential(
                nn.Linear(self.hilbert.size, hidden_dim, dtype=self.param_dtype),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.Nf+self.Nh, dtype=self.param_dtype)
            ))

        # Convert NNs to the appropriate data type
        for i in range(self.Nh):
            getattr(self, f'nn{i}').to(self.param_dtype)
        
        if self.jastrow:
            self.nn_jastrow = nn.Sequential(
                nn.Linear(self.hilbert.size, hidden_dim, dtype=self.param_dtype),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, dtype=self.param_dtype)
            )
            self.nn_jastrow.to(self.param_dtype)
        else:
            self.nn_jastrow = None

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'HFDS':{'N_site': self.hilbert.size, 'N_fermions': self.hilbert.n_fermions, 'N_fermions_per_spin': self.hilbert.n_fermions_per_spin,
                    'N_hidden_fermions': self.Nh, 'Jastrow': self.jastrow}
        }

    def amplitude(self, x):
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        # Define the slater determinant function manually to loop over inputs
        def backflow_det(n):
            # Compute the hidden fermion rows using the neural networks
            F = []
            for i in range(self.Nh):
                F.append(getattr(self, f'nn{i}')(n))
            F = torch.stack(F)

            M  = self.M
            # Find the positions of the occupied orbitals
            R = torch.nonzero(n, as_tuple=False).squeeze()

            # Extract the Nf x (Nf+Nh) submatrix of M corresponding to the occupied orbitals
            A = M[R]

            # Append the hidden fermion rows to corresponding A matrix
            Au = torch.cat((A, F), dim=0) # Shape: (Nf+Nh, Nf+Nh)
            det = torch.linalg.det(Au)
            amp = det

            return amp
        
        batch_amps = []
        for x_i in x:
            n_i = from_quimb_config_to_netket_config(x_i)
            # Check x_i type
            if not type(n_i) == torch.Tensor:
                n_i = torch.tensor(n_i, dtype=self.param_dtype)

            amp_val=backflow_det(n_i)
            if self.jastrow:
                J = torch.sum(self.nn_jastrow(n_i))
                amp_val *= torch.exp(J)
            batch_amps.append(amp_val)
        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class FFNN(wavefunctionModel):
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64):
        super(FFNN, self).__init__()
        
        self.hilbert = hilbert
        self.param_dtype = param_dtype

        # Initialize the neural network layer, input is n and output a matrix with the same shape as M
        self.nn = nn.Sequential(
            nn.Linear(self.hilbert.n_orbitals, hidden_dim, dtype=self.param_dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, dtype=self.param_dtype)
        )
        self.nn.to(self.param_dtype)

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'FFNN':{'N_orbitals': self.hilbert.n_orbitals, 'N_fermions': self.hilbert.n_fermions}
        }

    def forward(self, n):
        if not type(n) == torch.Tensor:
            n = torch.tensor(n, dtype=self.param_dtype)
        n = n.to(self.param_dtype)
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
