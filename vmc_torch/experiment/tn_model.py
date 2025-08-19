from mpi4py import MPI
import ast
import time
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap

# quimb
import quimb.tensor as qtn
from quimb.tensor.tensor_2d import Rotator2D, pairwise
import cotengra as ctg

from vmc_torch.fermion_utils import insert_proj_peps, flatten_proj_params, reconstruct_proj_params, insert_compressor
from vmc_torch.fermion_utils import calculate_phase_from_adjacent_trans_dict, decompose_permutation_into_transpositions
from vmc_torch.global_var import DEBUG, set_debug
from .nn_sublayers import *
flatten_tn_params = flatten_proj_params
reconstruct_tn_params = reconstruct_proj_params

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


class wavefunctionModel(torch.nn.Module):
    """Common class functions for all fermionic VMC models"""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        self.cache_env_mode = False
    
    def from_params_to_vec(self):
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)
    
    def clear_wavefunction_env_cache(self):
        pass
    
    def update_env_x_cache_to_row(self, *args, **kwargs):
        pass
    def update_env_y_cache_to_col(self, *args, **kwargs):
        pass
    def clear_env_x_cache(self):
        pass
    def clear_env_y_cache(self):
        pass

    def update_cached_cache(self):
        pass

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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
    def load_params(self, vec):
        pointer = 0
        for param in self.parameters():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            with torch.no_grad():
                param.copy_(new_param_values)
            pointer += num_param
    
    def from_vec_to_params(self, vec):
        # XXX: useful at all? Yes for spin PEPS
        pointer = 0
        new_params = {}
        for tid, param in self.torch_tn_params.items():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            new_params[tid] = new_param_values
            pointer += num_param
        return new_params
    
    @property
    def num_params(self):
        return len(self.from_params_to_vec())
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x, **kwargs)

class TN_backflow_attn_Tensorwise_Model_v1(wavefunctionModel):
    """
        For each on-site fermionic tensor with specific shape, assign a narrow on-site projector MLP with corresponding output dimension.
        This is to avoid the large number of parameters in the previous model, where Np = N_neurons * N_TNS.
    """
    def __init__(self, peps, max_bond=None, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(peps)

        self.torch_tn_params = {
            str(tid): nn.Parameter(data)  # Convert Tensor to Parameter
            for tid, data in params.items()
        }
        # register the torch tensors as parameters
        for tid, param in self.torch_tn_params.items():
            self.register_parameter(tid, param)

        # Define the neural network
        input_dim = peps.Lx * peps.Ly
        phys_dim = peps.phys_dim()
        
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
            mlp_input_dim = peps.Lx * peps.Ly * embedding_dim
            tn_params_vec = params[int(tid)].reshape(-1)  # flatten the tensor to get the vector form
            self.mlp[tid] = nn.Sequential(
                nn.Linear(mlp_input_dim, nn_final_dim),
                nn.LeakyReLU(),
                nn.Linear(nn_final_dim, tn_params_vec.numel()),
            )
            self.mlp[tid].to(self.param_dtype)

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'PEPS_backflow_attn_Tensorwise':
            {
                'D': peps.max_bond(), 
                'Lx': peps.Lx, 'Ly': peps.Ly, 
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
    
    def from_vec_to_params(self, vec):
        # XXX: useful at all? Yes for spin PEPS
        pointer = 0
        new_params = {}
        for tid, param in self.torch_tn_params.items():
            num_param = param.numel()
            new_param_values = vec[pointer:pointer+num_param].view(param.shape)
            new_params[int(tid)] = new_param_values
            pointer += num_param
        return new_params
    
    def amplitude(self, x):
        tn_params_vec = torch.cat(
            [param.reshape(-1) for param in self.torch_tn_params.values()]
        )

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
            tn_nn_params_vec = tn_params_vec + self.nn_eta*nn_correction
            # Ensure the new parameters are in the correct format for the TN
            tn_nn_params = self.from_vec_to_params(tn_nn_params_vec)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.isel({psi.site_inds[i]: int(s) for i, s in enumerate(x_i)})

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
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
        return torch.cat([param.reshape(-1) for param in two_row_params.values()])
    
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def forward(self, x, **kwargs):
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
        return torch.cat([param.reshape(-1) for param in tn_params.values()])
    
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            if param is not None:
                param.grad = None
    
    def forward(self, x, **kwargs):
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)



#------------ fermionic TN model with NN backflow-type correction ------------

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

class fMPS_backflow_attn_Tensorwise_Model_v1(wavefunctionModel):
    def __init__(self, ftn, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=torch.float32):
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

        # Define the shared attention block
        input_dim = ftn.L
        phys_dim = ftn.phys_dim()
        self.attention_block = SelfAttn_block(
            n_site=input_dim,
            num_classes=phys_dim,
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            dtype=self.param_dtype
        )

        # Define site-wise MLPs
        self.mlp = nn.ModuleDict()
        for tid in self.torch_tn_params.keys():
            tn_params_dict = {
                tid: params[int(tid)]
            }
            tn_params_vec = flatten_tn_params(tn_params_dict)
            self.mlp[tid] = nn.Sequential(
                nn.Linear(input_dim * embedding_dim, nn_final_dim),
                nn.LeakyReLU(),
                nn.Linear(nn_final_dim, tn_params_vec.numel()),
            )
            self.mlp[tid].to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fMPS_backflow_attn_Tensorwise': {
                'D': ftn.max_bond(),
                'L': ftn.L,
                'symmetry': self.symmetry,
                'nn_final_dim': nn_final_dim,
                'nn_eta': nn_eta,
                'embedding_dim': embedding_dim,
                'attention_heads': attention_heads,
            },
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
        
            # Get the shared attention features
            nn_features = self.attention_block(x_i)
            nn_features_vec = nn_features.view(-1)
            
            # Get the NN correction to the parameters, concatenate the results for each tensor
            nn_correction = torch.cat([self.mlp[tid](nn_features_vec) for tid in self.torch_tn_params.keys()])
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta * nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            amp_val = amp.contract()
            if amp_val == 0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

# define Receptive field for each site to its neighboring sites within radius r
# for sites on the boundary, the receptive field is smaller due to open boundary condition
def get_receptive_field_1d(L, r):
    receptive_field = {}
    for i in range(L):
        neighbors = []
        for j in range(max(0, i - r), min(L, i + r + 1)):
            neighbors.append(j)
        receptive_field[i] = neighbors
    return receptive_field

class fMPS_BFA_cluster_Model(wavefunctionModel):
    """fMPS+NN model with a NN receptive field of radius r for each site"""
    def __init__(self, fmps, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, radius=1, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(fmps)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get the receptive field for each site
        self.receptive_field = get_receptive_field_1d(fmps.L, r=radius)

        phys_dim = fmps.phys_dim()
        # for each tensor (labelled by tid), assign a attention+MLP
        self.nn = nn.ModuleDict()
        for tid in self.torch_tn_params.keys():
            # get the receptive field for the current tensor
            neighbors = self.receptive_field[int(tid)]
            input_dim = len(neighbors)
            on_site_ts_params_dict ={
                tid: params[int(tid)]
            }
            on_site_ts_params_vec = flatten_tn_params(on_site_ts_params_dict)
            self.nn[tid] = SelfAttn_FFNN_block(
                n_site=input_dim,
                num_classes=phys_dim,
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                nn_hidden_dim=nn_final_dim,
                output_dim=on_site_ts_params_vec.numel(),
                dtype=self.param_dtype
            )

        # Get symmetry
        self.symmetry = fmps.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fMPS_BFA_cluster': {
                'D': fmps.max_bond(),
                'L': fmps.L,
                'radius': radius,
                'symmetry': self.symmetry,
                'nn_final_dim': nn_final_dim,
                'nn_eta': nn_eta,
                'embedding_dim': embedding_dim,
                'attention_heads': attention_heads,
            },
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
            
            # For each site get the corresponding input config from x_i and receptive field
            local_configs = {tid: x_i[neighbors] for tid, neighbors in self.receptive_field.items()}
            # Get the corresponding NN output for each site and concatenate
            nn_correction = torch.cat([self.nn[tid](local_configs[int(tid)]) for tid in self.torch_tn_params.keys()])
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta * nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            amp_val = amp.contract()
            if amp_val == 0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class fMPS_BFA_cluster_Model_reuse(wavefunctionModel):
    """fMPS+NN model with a NN receptive field of radius r for each site"""
    def __init__(self, fmps, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, radius=1, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(fmps)

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get the receptive field for each site
        self.receptive_field = get_receptive_field_1d(fmps.L, r=radius)

        phys_dim = fmps.phys_dim()
        # for each tensor (labelled by tid), assign a attention+MLP
        self.nn = nn.ModuleDict()
        for tid in self.torch_tn_params.keys():
            # get the receptive field for the current tensor
            neighbors = self.receptive_field[int(tid)]
            input_dim = len(neighbors)
            on_site_ts_params_dict ={
                tid: params[int(tid)]
            }
            on_site_ts_params_vec = flatten_tn_params(on_site_ts_params_dict)
            self.nn[tid] = SelfAttn_FFNN_block(
                n_site=input_dim,
                num_classes=phys_dim,
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                nn_hidden_dim=nn_final_dim,
                output_dim=on_site_ts_params_vec.numel(),
                dtype=self.param_dtype
            )

        # Get symmetry
        self.symmetry = fmps.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fMPS_BFA_cluster_reuse': {
                'D': fmps.max_bond(),
                'L': fmps.L,
                'radius': radius,
                'symmetry': self.symmetry,
                'nn_final_dim': nn_final_dim,
                'nn_eta': nn_eta,
                'embedding_dim': embedding_dim,
                'attention_heads': attention_heads,
            },
        }
        self.nn_eta = nn_eta
        self.config_ref = None
        self.nn_output_cache = {}
        self.amp_tn_cache = None
        self.new_nn_output_cache = {}
        self.new_amp_tn_cache = None
        self.new_config_ref = None
        # always make sure the 3 nos are the same, when ever setting one, set the others
        self.config_ref_no = self.nn_output_cache_no = self.amp_tn_cache_no = 0
        self._env_left_cache = None
        self._env_right_cache = None
        self.nn_radius = radius
        self.L = fmps.L
    
    def transform_quimb_env_left_key_to_config_key(self, env_left, config):
        """
            Return a dictionary with the keys being the configs to the left of site i in quimb env_left.
            env_left: dict({site: tensor}), where the tensor is the left environment of the site.
        """
        env_left_config = {}
        for site in env_left.keys():
            if site != 0:
                config_key = tuple(config[:site].to(torch.int).tolist())
                env_left_config[config_key] = env_left[site]
        return env_left_config
    
    def transform_quimb_env_right_key_to_config_key(self, env_right, config):
        """
            Return a dictionary with the keys being the configs to the right of site i in quimb env_right.
            env_right: dict({site: tensor}), where the tensor is the right environment of the site.
        """
        env_right_config = {}
        for site in env_right.keys():
            if site != self.L - 1:
                config_key = tuple(config[site+1:].to(torch.int).tolist())
                env_right_config[config_key] = env_right[site]
        return env_right_config
    
    def cache_env_left(self, amp, config):
        """
            Cache the left environment of the TN.
            amp: quimb tensor network object.
            config: the configuration of the TN.
        """
        env_left = amp.compute_left_environments()
        env_left_config = self.transform_quimb_env_left_key_to_config_key(env_left, config)
        self._env_left_cache = env_left_config
        self.set_config_ref(config)

    def cache_env_right(self, amp, config):
        """
            Cache the right environment of the TN.
            amp: quimb tensor network object.
            config: the configuration of the TN.
        """
        env_right = amp.compute_right_environments()
        env_right_config = self.transform_quimb_env_right_key_to_config_key(env_right, config)
        self._env_right_cache = env_right_config
        self.set_config_ref(config)
    
    def cache_env(self, amp, config):
        """
            Cache the left and right environment of the TN.
            amp: quimb tensor network object.
            config: the configuration of the TN.
        """
        self.cache_env_left(amp, config)
        self.cache_env_right(amp, config)
        assert (self.config_ref == config).all(), "The cached environment does not match the current configuration."
    
    @property
    def env_left_cache(self):
        """
            Return the left environment of the TN.
        """
        if hasattr(self, '_env_left_cache'):
            return self._env_left_cache
        else:
            raise ValueError("The left environment is not cached. Please call cache_env_left() first.")
        
    @property
    def env_right_cache(self):
        """
            Return the right environment of the TN.
        """
        if hasattr(self, '_env_right_cache'):
            return self._env_right_cache
        else:
            raise ValueError("The right environment is not cached. Please call cache_env_right() first.")
        
    def clear_env_left_cache(self):
        """
            Clear the left environment cache.
        """
        self._env_left_cache = None
    def clear_env_right_cache(self):
        """
            Clear the right environment cache.
        """
        self._env_right_cache = None
    
    def clear_wavefunction_env_cache(self):
        self.clear_env_left_cache()
        self.clear_env_right_cache()
        self.nn_output_cache = {}
        self.amp_tn_cache = None
        self.config_ref = None
        self.amp_tn_cache_no = 0
        self.nn_output_cache_no = 0
        self.config_ref_no = 0

        self.new_nn_output_cache = {}
        self.new_amp_tn_cache = None
        self.new_config_ref = None

    
    def set_config_ref(self, config):
        """
            Set the reference configuration of the TN.
            config: the configuration of the TN.
        """
        self.config_ref = config
        self.config_ref_no = hash(tuple(config.to(torch.int).tolist()))
    
    def set_nn_output_cache(self, nn_output_cache):
        """
            Set the NN output cache.
            nn_output_cache: the NN output cache.
        """
        self.nn_output_cache = nn_output_cache
        self.nn_output_cache_no = hash(tuple(self.config_ref.to(torch.int).tolist()))
    
    def set_amp_tn_cache(self, amp_tn_cache):
        """
            Set the amplitude TN cache.
            amp_tn_cache: the amplitude TN cache.
        """
        self.amp_tn_cache = amp_tn_cache
        self.amp_tn_cache_no = hash(tuple(self.config_ref.to(torch.int).tolist()))
    
    def update_cached_cache(self):
        self.set_config_ref(self.new_config_ref)
        self.set_nn_output_cache(self.new_nn_output_cache)
        self.set_amp_tn_cache(self.new_amp_tn_cache)
    
    def detect_effected_sites(self, config_ref, new_config):
        effected_sites = set()
        for i in range(self.L):
            if not torch.equal(config_ref[i], new_config[i]):
                effected_sites_left = max(0, i - self.nn_radius)
                effected_sites_right = min(self.L - 1, i + self.nn_radius)
                effected_sites.update(list(range(effected_sites_left, effected_sites_right + 1)))
        
        effected_sites = sorted(effected_sites)
        
        if len(effected_sites) == 0:
            return [], [], []
        
        uneffected_sites_left = list(range(effected_sites[0]))
        uneffected_sites_right = list(range(effected_sites[-1] + 1, self.L))

        return effected_sites, uneffected_sites_left, uneffected_sites_right

    def detect_changed_sites(self, config_ref, new_config):
        """
            Detect the sites that have changed in the new configuration.
            config_ref: the reference configuration of the TN.
            new_config: the new configuration of the TN.
        """
        changed_sites = set()
        for i in range(self.L):
            if not torch.equal(config_ref[i], new_config[i]):
                changed_sites.add(i)
        
        changed_sites = sorted(changed_sites)
        if len(changed_sites) == 0:
            return [], [], []
        uneffected_sites_left = list(range(changed_sites[0]))
        uneffected_sites_right = list(range(changed_sites[-1] + 1, self.L))
        return changed_sites, uneffected_sites_left, uneffected_sites_right
    
    def get_local_amp_tensors(self, tids:list, config:torch.Tensor, nn_output_cache: Optional[dict] = None):
        """
            Get the local tensors for the given tensor ids and configuration.
            tids: a list of tensor ids. list of int.
            config: the input configuration.
        """
        # first pick out the tensor parameters and form the local tn parameters vector
        local_ts_params = {}
        for tid in tids:
            local_ts_params[tid] = {
                ast.literal_eval(sector): data
                for sector, data in self.torch_tn_params[str(tid)].items()
            }
        local_ts_params_vec = flatten_tn_params(local_ts_params)

        # then select the corresponding NN outputs for the given tensor ids
        local_nn_correction = torch.cat([nn_output_cache[tid] for tid in tids])
        # Add the correction to the original parameters
        local_tn_nn_params = reconstruct_proj_params(local_ts_params_vec + self.nn_eta * local_nn_correction, local_ts_params)

        # Select the corresponding tensor skeleton
        local_ts_skeleton = self.skeleton.select([self.skeleton.site_tag_id.format(tid) for tid in tids], which='any')

        # Reconstruct the TN with the new parameters
        local_ts = qtn.unpack(local_tn_nn_params, local_ts_skeleton)

        # Fix the physical indices
        return local_ts.fix_phys_inds(tids, config[tids])

    def get_amp_tn(self, config, params=None, params_vec=None, cache_nn_output=False, cache_amp_tn=False):
        if self.nn_output_cache == {}:
            nn_output_cache = {tid: self.nn[str(tid)](config[neighbors]) for tid, neighbors in self.receptive_field.items()}
            nn_correction = torch.cat(list(nn_output_cache.values()))
            if cache_nn_output:
                self.set_nn_output_cache(nn_output_cache)
                self.new_nn_output_cache = self.nn_output_cache.copy()
                assert self.nn_output_cache_no == self.config_ref_no, "The nn_output_cache_no does not match the config_ref_no."
        # reuse cached NN outputs for config_ref
        else:
            effected_sites, _, _ = self.detect_effected_sites(self.config_ref, config)
            new_nn_output_cache = self.nn_output_cache.copy()
            for tid in effected_sites:
                neighbors = self.receptive_field[tid]
                # get the new input config for the current tensor
                config[neighbors] = config[neighbors]
                # get the new NN output for the current tensor
                new_nn_output_cache[tid] = self.nn[str(tid)](config[neighbors])

            self.new_nn_output_cache = new_nn_output_cache

            # get the new NN output for the current tensor
            nn_correction = torch.cat(list(new_nn_output_cache.values()))
        
        # Get the amplitude tn, potentially using cached amp tensors
        if self.amp_tn_cache is None:
            if params is None and params_vec is None:
                # Reconstruct the original parameter structure (by unpacking from the flattened dict)
                params = {
                    int(tid): {
                        ast.literal_eval(sector): data
                        for sector, data in blk_array.items()
                    }
                    for tid, blk_array in self.torch_tn_params.items()
                }
                params_vec = flatten_tn_params(params)
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta * nn_correction, params)
            # Reconstruct the TN with the new parameters
            fmps = qtn.unpack(tn_nn_params, self.skeleton)
            amp = fmps.get_amp(config)
            if cache_amp_tn:
                self.set_amp_tn_cache(amp)
                self.new_amp_tn_cache = amp.copy()
                assert self.amp_tn_cache_no == self.config_ref_no, "The amp_tn_cache_no does not match the config_ref_no."
        else:
            if self.nn_output_cache == {}:
                raise ValueError("The nn_output_cache is empty. Please call get_amp_tn() with cache_nn_output=True first.")
            effected_sites, uneffected_sites_left, uneffected_sites_right = self.detect_effected_sites(self.config_ref, config)
            if len(effected_sites) != 0:
                effected_ts = self.get_local_amp_tensors(effected_sites, config, self.nn_output_cache)
                uneffected_sites_left_tags = [self.skeleton.site_tag_id.format(tid) for tid in uneffected_sites_left]
                uneffected_sites_right_tags = [self.skeleton.site_tag_id.format(tid) for tid in uneffected_sites_right]
                uneffected_tn_left = self.amp_tn_cache.select(uneffected_sites_left_tags, which='any')
                uneffected_tn_right = self.amp_tn_cache.select(uneffected_sites_right_tags, which='any')
                amp = uneffected_tn_left | uneffected_tn_right | effected_ts # updated amp tn
            else:
                amp = self.amp_tn_cache
            
            self.new_amp_tn_cache = amp
        
        self.new_config_ref = config

        return amp
    
    def update_env_left(self, config):
        if self._env_left_cache is not None:
            self.clear_env_left_cache()
        amp_tn = self.get_amp_tn(config)
        self.cache_env_left(amp_tn, config)
        assert (self.config_ref == config).all(), "The cached environment does not match the current configuration."

    def update_env_right(self, config):
        if self._env_right_cache is not None:
            self.clear_env_right_cache()
        amp_tn = self.get_amp_tn(config)
        self.cache_env_right(amp_tn, config)
        assert (self.config_ref == config).all(), "The cached environment does not match the current configuration."
    
    def update_env_to_site(self, amp_tn, config, site_id, from_which='left'):
        # t0 = MPI.Wtime()
        left_site_id = max(0, site_id - self.nn_radius)
        right_site_id = min(self.L - 1, site_id + self.nn_radius)

        # select the site ts
        site_id = left_site_id if from_which == 'left' else right_site_id
        site_ts = amp_tn.select(site_id)

        if from_which == 'left':
            key_prev_sites = tuple(config[:site_id].to(torch.int).tolist()) if site_id > 0 else ()
            new_env_key = tuple(config[:site_id+1].to(torch.int).tolist())
        elif from_which == 'right':
            key_prev_sites = tuple(config[site_id+1:].to(torch.int).tolist()) if site_id < self.L - 1 else ()
            new_env_key = tuple(config[site_id:].to(torch.int).tolist())
        else:
            raise ValueError("from_which must be either 'left' or 'right'.")

        if from_which == 'left':
            if self.env_left_cache is not None:
                if key_prev_sites in self.env_left_cache:
                    # print('use cached env to update cache')
                    prev_env_left = self.env_left_cache[key_prev_sites]
                    new_env_ts = (site_ts|prev_env_left).contract()
                    new_env_left_cache = {new_env_key: new_env_ts}
                    # t1 = MPI.Wtime()
                    # print(f'Update cache time0={t1-t0}')
                else:
                    # raise NotImplementedError("Left environment cache is not implemented yet.")
                    # quimb left env calculation
                    left_envs = {1: amp_tn.select(0).contract()}
                    for i in range(2, site_id + 1):
                        tll = left_envs[i - 1]
                        tll.drop_tags()
                        tnl = amp_tn.select(i - 1) | tll
                        left_envs[i] = tnl.contract()
                    new_env_left_cache = self.transform_quimb_env_left_key_to_config_key(left_envs, config)
                self._env_left_cache.update(new_env_left_cache)
            else:
                # quimb left env calculation
                left_envs = {1: amp_tn.select(0).contract()}
                for i in range(2, site_id + 1):
                    tll = left_envs[i - 1]
                    tll.drop_tags()
                    tnl = amp_tn.select(i - 1) | tll
                    left_envs[i] = tnl.contract()
                new_env_left_cache = self.transform_quimb_env_left_key_to_config_key(left_envs, config)
                self._env_left_cache = new_env_left_cache

        elif from_which == 'right':
            if self.env_right_cache is not None:
                if key_prev_sites in self.env_right_cache:
                    prev_env_right = self.env_right_cache[key_prev_sites]
                    new_env_ts = (site_ts|prev_env_right).contract()
                    new_env_right_cache = {new_env_key: new_env_ts}
                else:
                    # print(config, site_id, key_prev_sites)
                    # raise NotImplementedError("Right environment cache is not implemented yet.")
                    # quimb right env calculation
                    right_envs = {self.L - 2: amp_tn.select(-1).contract()}
                    for i in range(self.L - 3, site_id - 1, -1):
                        trl = right_envs[i + 1]
                        trl.drop_tags()
                        trn = amp_tn.select(i + 1) | trl
                        right_envs[i] = trn.contract()
                    new_env_right_cache = self.transform_quimb_env_right_key_to_config_key(right_envs, config)
                
                self._env_right_cache.update(new_env_right_cache)
            else:
                # quimb right env calculation
                right_envs = {self.L - 2: amp_tn.select(-1).contract()}
                for i in range(self.L - 3, site_id - 1, -1):
                    trl = right_envs[i + 1]
                    trl.drop_tags()
                    trn = amp_tn.select(i + 1) | trl
                    right_envs[i] = trn.contract()
                new_env_right_cache = self.transform_quimb_env_right_key_to_config_key(right_envs, config)
                self._env_right_cache = new_env_right_cache
        else:
            raise ValueError("from_which must be either 'left' or 'right'.")
        
        self.set_config_ref(config)
    
    def amplitude(self, x):
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)

            if self.cache_env_mode:
                self.clear_wavefunction_env_cache()
                self.config_ref = x_i
                self.config_ref_no = hash(tuple(x_i.to(torch.int).tolist()))
                amp = self.get_amp_tn(x_i, params=params, params_vec=params_vec, cache_nn_output=True, cache_amp_tn=True)
                self.cache_env(amp, x_i)
                key_left = tuple(x_i[:self.L//2].to(torch.int).tolist())
                key_right = tuple(x_i[self.L//2:].to(torch.int).tolist())
                amp_left = self.env_left_cache[key_left]
                amp_right = self.env_right_cache[key_right]
                amp_val = (amp_left | amp_right).contract()
                # t0 = MPI.Wtime()
                # amp_val.backward(retain_graph=True)
                # t1 = MPI.Wtime()
                # amp.contract().backward(retain_graph=True)
                # t2 = MPI.Wtime()
                # print(f'Backward time reuse: {t1-t0}, backward time no reuse: {t2-t1}')
                
            else:
                amp = self.get_amp_tn(x_i, params=params, params_vec=params_vec, cache_nn_output=False, cache_amp_tn=False)
                if self.env_left_cache is None and self.env_right_cache is None:
                    print('No cache, exact contraction')
                    amp_val = amp.contract()
                else:
                    effected_sites, uneffected_sites_left, uneffected_sites_right = self.detect_effected_sites(self.config_ref, x_i)
                    changed_sites, _, _ = self.detect_changed_sites(self.config_ref, x_i)
                    if len(changed_sites) == 0:
                        # If no sites are changed, use the cached environment
                        amp_val = (self.env_left_cache[tuple(x_i[:self.L//2].to(torch.int).tolist())] | self.env_right_cache[tuple(x_i[self.L//2:].to(torch.int).tolist())]).contract()
                    else:
                        effected_sites_tag = [amp.site_tag_id.format(i) for i in effected_sites]
                        effected_tn = amp.select(effected_sites_tag, which='any')
                        left_env = qtn.TensorNetwork()
                        right_env = qtn.TensorNetwork()

                        if len(uneffected_sites_left) > 0:
                            left_env = self.env_left_cache[tuple(x_i[uneffected_sites_left].to(torch.int).tolist())]
                        if len(uneffected_sites_right) > 0:
                            right_env = self.env_right_cache[tuple(x_i[uneffected_sites_right].to(torch.int).tolist())]
                        amp_val = (effected_tn | left_env | right_env).contract()

            if amp_val == 0.0:
                amp_val = torch.tensor(0.0)
            
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def amplitude_n_tn(self, x, cache=None, cache_nn_output=False, cache_amp_tn=False, initial_cache=False):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)

        batch_amps = []
        assert x.size(0) == 1, "x must be a batch of size 1."
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(self.param_dtype)
            
            if initial_cache:
                self.clear_wavefunction_env_cache()
                self.set_config_ref(x_i)
            
            amp = self.get_amp_tn(
                x_i, 
                cache_nn_output=cache_nn_output, 
                cache_amp_tn=cache_amp_tn, 
            )
            
            key_left = tuple(x_i[:self.L-self.nn_radius].to(torch.int).tolist())
            key_right = tuple(x_i[self.nn_radius:].to(torch.int).tolist())

            if cache == 'left': # change config_ref
                self.cache_env_left(amp, x_i)
                uncontracted_right_tag = [amp.site_tag_id.format(i) for i in range(self.L-self.nn_radius, self.L)]
                amp_uncontracted_right = amp.select(uncontracted_right_tag, which='any')
                amp_val = (self.env_left_cache[key_left] | amp_uncontracted_right).contract()
            elif cache == 'right': # change config_ref
                self.cache_env_right(amp, x_i)
                uncontracted_left_tag = [amp.site_tag_id.format(i) for i in range(0, self.nn_radius)]
                amp_uncontracted_left = amp.select(uncontracted_left_tag, which='any')
                amp_val = (self.env_right_cache[key_right] | amp_uncontracted_left).contract()
            elif cache == None: # do not change config_ref
                if self.env_left_cache is None and self.env_right_cache is None:
                    print('No cache, exact contraction')
                    amp_val = amp.contract()
                else:
                    effected_sites, uneffected_sites_left, uneffected_sites_right = self.detect_effected_sites(self.config_ref, x_i)
                    changed_sites, _, _ = self.detect_changed_sites(self.config_ref, x_i)
                    if len(changed_sites) == 0:
                        # If no sites are changed, use the cached environment
                        amp_val = (self.env_left_cache[tuple(x_i[:self.L//2].to(torch.int).tolist())] | self.env_right_cache[tuple(x_i[self.L//2:].to(torch.int).tolist())]).contract()
                    else:
                        effected_sites_tag = [amp.site_tag_id.format(i) for i in effected_sites]
                        effected_tn = amp.select(effected_sites_tag, which='any')
                        left_env = qtn.TensorNetwork()
                        right_env = qtn.TensorNetwork()

                        if len(uneffected_sites_left) > 0:
                            left_env = self.env_left_cache[tuple(x_i[uneffected_sites_left].to(torch.int).tolist())]
                        if len(uneffected_sites_right) > 0:
                            right_env = self.env_right_cache[tuple(x_i[uneffected_sites_right].to(torch.int).tolist())]
                        amp_val = (effected_tn | left_env | right_env).contract()

            if amp_val == 0.0:
                amp_val = torch.tensor(0.0, device=x.device, dtype=self.param_dtype)
            
            batch_amps.append(amp_val)
        
        return torch.stack(batch_amps), amp
            
            

#------------ fermionic PEPS based model ------------

class fTNModel(wavefunctionModel):

    def __init__(self, ftn, max_bond=None, dtype=torch.float32, functional=False):
        super().__init__()
        self.param_dtype = dtype
        self.functional = functional
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
    
    def get_amp_tn(self, config):
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
        # Get the amplitude
        amp = psi.get_amp(config)

        return amp

    
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
                x_i = torch.tensor(x_i, dtype=torch.int if self.functional else self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(torch.int if self.functional else self.param_dtype)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True, functional=self.functional)
            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree)

            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                amp_val = amp.contract()

            if not self.functional:
                if amp_val==0.0:
                    amp_val = torch.tensor(0.0, device=x.device)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class fTNModel_Jastrow(wavefunctionModel):

    def __init__(self, ftn, max_bond=None, dtype=torch.float32, nn_jastrow_hidden_dim=64, embedding_dim=16, attention_heads=4, exponential=False):
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

        self.attn = SelfAttn_block(
            n_site=ftn.Lx * ftn.Ly,
            num_classes=4,
            embedding_dim=embedding_dim,
            attention_heads=attention_heads,
            dtype=self.param_dtype,
        )
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads

        if not exponential:
            self.nn_jastrow = nn.Sequential(
                nn.Linear(ftn.Lx * ftn.Ly * embedding_dim, nn_jastrow_hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(nn_jastrow_hidden_dim, nn_jastrow_hidden_dim),
                ShiftedSinhYFixed()
            )
        else:
            self.nn_jastrow = nn.Sequential(
                nn.Linear(ftn.Lx * ftn.Ly * embedding_dim, nn_jastrow_hidden_dim),
                nn.GELU(),
                nn.Linear(nn_jastrow_hidden_dim, nn_jastrow_hidden_dim),
                nn.GELU(),
            )
        self.jastrow_exponential = exponential

        self.nn_jastrow = self.nn_jastrow.to(dtype)
        self.nn_jastrow_hidden_dim = nn_jastrow_hidden_dim

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            f'fPEPS (chi={max_bond})':{'D': ftn.max_bond(), 'Lx': ftn.Lx, 'Ly': ftn.Ly, 'symmetry': self.symmetry, 'nn_jastrow_hidden_dim': nn_jastrow_hidden_dim,
                                       'embedding_dim': embedding_dim, 'attention_heads': attention_heads},
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
                x_i = torch.tensor(x_i, dtype=torch.int if self.functional else self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(torch.int if self.functional else self.param_dtype)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)
            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree)

            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                amp_val = amp.contract()

            if amp_val==0.0:
                amp_val = torch.tensor(0.0, device=x.device)
            
            nn_jastrow = torch.mean(self.nn_jastrow(self.attn(x_i).view(-1))) if not self.jastrow_exponential else torch.exp(torch.mean(self.nn_jastrow(self.attn(x_i).view(-1))))

            amp_val = amp_val * nn_jastrow

            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class fTNModel_vec(wavefunctionModel):
    """Attempts to vectorize the fTNModel for faster computation."""
    def __init__(self, ftn, max_bond=None, dtype=torch.float32, functional=False, tree=None, device=None):
        super().__init__()
        self.param_dtype = dtype
        self.functional = functional
        self.device = device
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

        opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
        # Get the amplitude
        random_x = torch.randint(0, 3, (ftn.Lx*ftn.Ly,), dtype=torch.int)
        amp = ftn.get_amp(random_x, conj=True, functional=self.functional)
        self.tree = amp.contraction_tree(optimize=opt)

        # Get self parity
        self.parity_config = torch.tensor([array.parity for array in ftn.arrays], dtype=torch.int, device=self.device)

        # BUG: in tree.traverse(), the tids are not automatically sorted, so we need to sort them manually
        self.sorted_tree_traverse_path = {i: tuple(sorted(left_tids))+tuple(sorted(right_tids)) for i, (_, left_tids, right_tids) in enumerate(self.tree.traverse())}

        # compute the permutation dict for future global phase computation
        self.perm_dict = {i: tuple(torch.argsort(torch.tensor(left_right_tids))) for i, left_right_tids in self.sorted_tree_traverse_path.items()}
        self.perm_dict_desc = {i: tuple(torch.argsort(torch.tensor(tuple(sorted(left_tids))[::-1]+tuple(sorted(right_tids))[::-1]), descending=True)) for i, (_, left_tids, right_tids) in enumerate(self.tree.traverse())}
        
        self.adjacent_transposition_dict_asc = {i: decompose_permutation_into_transpositions(perm, asc=False) for i, perm in self.perm_dict.items()}
        self.adjacent_transposition_dict_desc = {i: decompose_permutation_into_transpositions(perm, asc=False) for i, perm in self.perm_dict_desc.items()}

    
    def compute_global_phase(self, input_config):
        """Get the global phase of contracting an amplitude of the fPEPS given computational graph."""
        on_site_parity_tensor = torch.tensor([0,1,1,0], dtype=torch.int, device=input_config.device)
        def get_parity(n):
            return on_site_parity_tensor[n]
        # input_parity_config = input_config % 2
        input_config_parity = get_parity(input_config)
        amp_parity_config = (self.parity_config + input_config_parity) % 2

        phase = 1
        phase *= calculate_phase_from_adjacent_trans_dict(
            self.tree, 
            input_config_parity, 
            self.parity_config, 
            amp_parity_config, 
            self.adjacent_transposition_dict_asc, 
            self.adjacent_transposition_dict_desc,
            self.sorted_tree_traverse_path
            )
            
        return phase

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

        def amplitude_func(psi, x_i):
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=torch.int if self.functional else self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(torch.int if self.functional else self.param_dtype)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True, functional=self.functional)
            if self.max_bond is None:
                amp = amp
                amp_val = amp.contract(optimize=self.tree)
                phase = self.compute_global_phase(x_i.int())
                amp_val = phase * amp_val

            else:
                amp = amp.contract_boundary_from_xmin(max_bond=self.max_bond, cutoff=0.0, xrange=[0, psi.Lx//2-1])
                amp = amp.contract_boundary_from_xmax(max_bond=self.max_bond, cutoff=0.0, xrange=[psi.Lx//2, psi.Lx-1])
                amp_val = amp.contract()

            # if amp_val==0.0:
            #     amp_val = torch.tensor(0.0)

            # Return the batch of amplitudes stacked as a tensor
            return amp_val
        
        # vec_amplitude_func = vmap(amplitude_func, in_dims=(None, 0), randomness='different')
        # # Get the amplitude
        # batch_amps = vec_amplitude_func(psi, x)

        return amplitude_func(psi, x)
    
    def forward(self, x, **kwargs):
        return self.amplitude(x)

class fTNModel_reuse(wavefunctionModel):
    def __init__(self, ftn, max_bond=None, dtype=torch.float32, functional=False, debug=False):
        super().__init__()
        self.param_dtype = dtype
        self.functional = functional
        self.debug = debug
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)
        # self.skeleton.exponent = 0

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        # NOTE: pytorch nn.ParameterDict automatically sorts the keys (as sorted(dict))
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })
        self.tn_param_key_id = 'torch_tn_params.{}.{}'

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
        self.Lx = ftn.Lx
        self.Ly = ftn.Ly
        self._env_x_cache = None
        self._env_y_cache = None
        self.config_ref = None
        self.amp_ref = None
        self.debug_amp_cache = []
    
    def from_1d_to_2d(self, config, ordering='snake'):
        if ordering == 'snake':
            config_2d = config.reshape((self.Lx, self.Ly))
            return config_2d
        else:
            raise NotImplementedError(f'Ordering {ordering} is not implemented.')
        
    def from_1dsite_to_2dsite(self, site, ordering='snake'):
        """
            Convert a 1d site index to a 2d site index.
            site: 1d site index
        """
        if ordering == 'snake':
            return (site // self.Ly, site % self.Ly)
        else:
            raise ValueError(f"Unsupported ordering: {ordering}")
    
    def from_2dsite_to_1dsite(self, site, ordering='snake'):
        """
            Convert a 2d site index to a 1d site index.
            site: (row, col) tuple
        """
        if ordering == 'snake':
            return site[0] * self.Ly + site[1]
        else:
            raise ValueError(f"Unsupported ordering: {ordering}")
    
    def transform_quimb_env_x_key_to_config_key(self, env_x, config):
        """
            Return a dictionary with the keys of of the config rows
        """
        config_2d = self.from_1d_to_2d(config)
        env_x_row_config = {}
        for key in env_x.keys():
            if key[0] == 'xmax': # from bottom to top
                row_n = key[1]
                if row_n != self.Lx-1:
                    rows_config = tuple(torch.cat(tuple(config_2d[row_n+1:].to(torch.int))).tolist())
                    env_x_row_config[('xmax', rows_config)] = env_x[key]
            elif key[0] == 'xmin': # from top to bottom
                row_n = key[1]
                if row_n != 0:
                    rows_config = tuple(torch.cat(tuple(config_2d[:row_n].to(torch.int))).tolist())
                    env_x_row_config[('xmin', rows_config)] = env_x[key]
        return env_x_row_config
    
    def transform_quimb_env_y_key_to_config_key(self, env_y, config):
        """
            Return a dictionary with the keys of of the config rows
        """
        config_2d = self.from_1d_to_2d(config)
        env_y_row_config = {}
        for key in env_y.keys():
            if key[0] == 'ymax':
                col_n = key[1]
                if col_n != self.Ly-1:
                    cols_config = tuple(torch.cat(tuple(config_2d[:, col_n+1:].to(torch.int))).tolist())
                    env_y_row_config[('ymax', cols_config)] = env_y[key]
            elif key[0] == 'ymin':
                col_n = key[1]
                if col_n != 0:
                    cols_config = tuple(torch.cat(tuple(config_2d[:, :col_n].to(torch.int))).tolist())
                    env_y_row_config[('ymin', cols_config)] = env_y[key]
        return env_y_row_config

    def cache_env_x(self, amp, config):
        """
            Cache the environment x for the given configuration
        """
        env_x = amp.compute_x_environments(max_bond=self.max_bond, cutoff=0.0)
        env_x_cache = self.transform_quimb_env_x_key_to_config_key(env_x, config)
        self._env_x_cache = env_x_cache
        self.config_ref = config
        self.amp_ref = amp
    
    def cache_env_y(self, amp, config):
        env_y = amp.compute_y_environments(max_bond=self.max_bond, cutoff=0.0)
        env_y_cache = self.transform_quimb_env_y_key_to_config_key(env_y, config)
        self._env_y_cache = env_y_cache
        self.config_ref = config
        self.amp_ref = amp
    
    def cache_env(self, amp, config):
        """
            Cache the environment x and y for the given configuration
        """
        self.cache_env_x(amp, config)
        self.cache_env_y(amp, config)
        
    @property
    def env_x_cache(self):
        """
            Return the cached environment x
        """
        if hasattr(self, '_env_x_cache'):
            return self._env_x_cache
        else:
            return None
        
    @property
    def env_y_cache(self):
        """
            Return the cached environment y
        """
        if hasattr(self, '_env_y_cache'):
            return self._env_y_cache
        else:
            return None
    
    def clear_env_x_cache(self):
        """
            Clear the cached environment x
        """
        self._env_x_cache = None

    def clear_env_y_cache(self):
        """
            Clear the cached environment y
        """
        self._env_y_cache = None
    
    def clear_wavefunction_env_cache(self):
        self.clear_env_x_cache()
        self.clear_env_y_cache()
        self.config_ref = None
        self.amp_ref = None
    
    def detect_changed_sites(self, config_ref, new_config):
        """
            Detect the sites that have changed in the new configuration,
            written in 1d coordinate format.
        """
        changed_sites = set()
        unchanged_sites = set()
        for i in range(self.Lx * self.Ly):
            if config_ref[i] != new_config[i]:
                changed_sites.add(i)
            else:
                unchanged_sites.add(i)
        changed_sites = sorted(changed_sites)
        unchanged_sites = sorted(unchanged_sites)
        if len(changed_sites) == 0:
            return [], []
        return changed_sites, unchanged_sites

    def from_1d_sites_to_tids(self, sites):
        """
            Convert a list of 1d site indices to a list of tensor ids.
        """
        tids_list = list(self.skeleton.tensor_map.keys())
        return [tids_list[site] for site in sites]
    
    def detect_changed_rows(self, config_ref, new_config):
        """
            Detect the rows that have changed in the new configuration
        """
        config_ref_2d = self.from_1d_to_2d(config_ref)
        new_config_2d = self.from_1d_to_2d(new_config)
        changed_rows = []
        for i in range(self.Lx):
            if not torch.equal(config_ref_2d[i], new_config_2d[i]):
                changed_rows.append(i)
        if len(changed_rows) == 0:
            return [], [], []
        unchanged_rows_above = list(range(changed_rows[0]))
        unchanged_rows_below = list(range(changed_rows[-1]+1, self.Lx))
        return changed_rows, unchanged_rows_above, unchanged_rows_below
    
    def detect_changed_cols(self, config_ref, new_config):
        """
            Detect the columns that have changed in the new configuration
        """
        config_ref_2d = self.from_1d_to_2d(config_ref)
        new_config_2d = self.from_1d_to_2d(new_config)
        changed_cols = []
        for i in range(self.Ly):
            if not torch.equal(config_ref_2d[:, i], new_config_2d[:, i]):
                changed_cols.append(i)
        if len(changed_cols) == 0:
            return [], [], []
        unchanged_cols_left = list(range(changed_cols[0]))
        unchanged_cols_right = list(range(changed_cols[-1]+1, self.Ly))
        return changed_cols, unchanged_cols_left, unchanged_cols_right
    
    def update_env_x_cache(self, config):
        """
            Update the cached environment x for the given configuration
        """
        if self.env_x_cache is not None:
            self.clear_env_x_cache()
        amp_tn = self.get_amp_tn(config)
        self.cache_env_x(amp_tn, config)
        self.config_ref = config
        self.amp_ref = amp_tn
    
    def update_env_x_cache_to_row(self, config, row_id, from_which='xmin'):
        amp_tn = self.get_amp_tn(config)
        new_env_x = amp_tn.compute_environments(max_bond=self.max_bond, cutoff=0.0, xrange=(0, row_id+1) if from_which=='xmin' else (row_id-1, self.Lx-1), from_which=from_which)
        new_env_x_cache = self.transform_quimb_env_x_key_to_config_key(new_env_x, config)
        # add the new env_x to the cache
        if self.env_x_cache is None:
            self._env_x_cache = new_env_x_cache
        else:
            self._env_x_cache.update(new_env_x_cache)
        self.config_ref = config
        self.amp_ref = amp_tn
    
    def update_env_y_cache(self, config):
        """
            Update the cached environment y for the given configuration
        """
        if self.env_y_cache is not None:
            self.clear_env_y_cache()
        amp_tn = self.get_amp_tn(config)
        self.cache_env_y(amp_tn, config)
        self.config_ref = config
        self.amp_ref = amp_tn
    
    def update_env_y_cache_to_col(self, config, col_id, from_which='ymin'):
        amp_tn = self.get_amp_tn(config)
        new_env_y = amp_tn.compute_environments(max_bond=self.max_bond, cutoff=0.0, yrange=(0, col_id+1) if from_which=='ymin' else (col_id-1, self.Ly-1), from_which=from_which)
        new_env_y_cache = self.transform_quimb_env_y_key_to_config_key(new_env_y, config)
        # add the new env_y to the cache
        if self.env_y_cache is None:
            self._env_y_cache = new_env_y_cache
        else:
            self._env_y_cache.update(new_env_y_cache)
        self.config_ref = config
        self.amp_ref = amp_tn
    
    def psi(self):
        """
            Return the wavefunction (fPEPS)
        """
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
        return psi

    def get_local_amp_tensors(self, sites:list, config:torch.Tensor):
        """
            Get the local tensors for the given tensor ids and configuration.
            tids: a list of tensor ids. list of int.
            config: the input configuration.
            sites: a list of 1d site/2d site indices corresponding to the tids.
        """
        # first pick out the tensor parameters and form the local tn parameters vector
        local_ts_params = {}
        # tids = self.from_1d_sites_to_tids(sites)
        tids = self.from_1d_sites_to_tids([self.from_2dsite_to_1dsite(site) for site in sites]) if isinstance(sites[0], tuple) else self.from_1d_sites_to_tids(sites)
        for tid in tids:
            local_ts_params[tid] = {
                ast.literal_eval(sector): data
                for sector, data in self.torch_tn_params[str(tid)].items()
            }
        
        # Get sites corresponding to the tids
        sites_1d = [self.from_2dsite_to_1dsite(site) for site in sites] if isinstance(sites[0], tuple) else sites
        sites_2d = sites if isinstance(sites[0], tuple) else [self.from_1dsite_to_2dsite(site) for site in sites]

        # Select the corresponding tensor skeleton
        local_ts_skeleton = self.skeleton.select([self.skeleton.site_tag_id.format(*site) for site in sites_2d], which='any')

        # Reconstruct the TN with the new parameters
        local_ftn = qtn.unpack(local_ts_params, local_ts_skeleton)

        # Fix the physical indices
        return local_ftn.fix_phys_inds(sites_2d, config[sites_1d])
    
    def get_amp_tn(self, config, reconstruct=False):

        if self.amp_ref is None or reconstruct:
            psi = self.psi()
            # Check config type
            if not type(config) == torch.Tensor:
                config = torch.tensor(config, dtype=torch.int if self.functional else self.param_dtype)
            else:
                if config.dtype != self.param_dtype:
                    config = config.to(torch.int if self.functional else self.param_dtype)
            # Get the amplitude
            amp_tn = psi.get_amp(config, conj=True, functional=self.functional)
            # if self.debug:
            #     print(f'Efficient amp tn construction (full), amp_tn exponent: {amp_tn.exponent}')
            return amp_tn
        
        else:
            # detect the sites that have changed
            changed_sites, unchanged_sites = self.detect_changed_sites(self.config_ref, config)

            if len(changed_sites) == 0:
                return self.amp_ref
            else:
                # substitute the changed sites tensors
                local_amp_tn = self.get_local_amp_tensors(changed_sites, config)
                unchanged_sites_2d = [self.from_1dsite_to_2dsite(site) for site in unchanged_sites]
                unchanged_sites_tags = [self.skeleton.site_tag_id.format(*site) for site in unchanged_sites_2d]
                unchanged_amp_tn = self.amp_ref.select(unchanged_sites_tags, which='any')
                # merge the local_amp_tn and unchanged_amp_tn
                amp_tn = local_amp_tn | unchanged_amp_tn
                amp_tn.exponent = self.skeleton.exponent
                # if self.debug:
                #     print(f'Efficient amp tn construction (local), amp_tn exponent: {amp_tn.exponent}')
                return amp_tn
    
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
                x_i = torch.tensor(x_i, dtype=torch.int if self.functional else self.param_dtype)
            else:
                if x_i.dtype != self.param_dtype:
                    x_i = x_i.to(torch.int if self.functional else self.param_dtype)
            # Get the amplitude
            # amp = psi.get_amp(x_i, conj=True, functional=self.functional)
            amp_tn = self.get_amp_tn(x_i)

            if self.max_bond is None:
                amp = amp_tn
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree) # quimb will address the cached exponent automatically

            else:
                if self.cache_env_mode:
                    self.cache_env_x(amp_tn, x_i)
                    # self.cache_env_y(amp, x_i)
                    self.config_ref = x_i
                    config_2d = self.from_1d_to_2d(x_i)
                    key_bot = ('xmax', tuple(torch.cat(tuple(config_2d[self.Lx//2:].to(torch.int))).tolist()))
                    key_top = ('xmin', tuple(torch.cat(tuple(config_2d[:self.Lx//2].to(torch.int))).tolist()))
                    amp_bot = self.env_x_cache[key_bot]
                    amp_top = self.env_x_cache[key_top]
                    amp_val = (amp_bot|amp_top).contract()*10**(self.skeleton.exponent) # quimb cannot address the cached exponent automatically when TN reuses the cached environment, so we need to multiply it manually
                    if self.debug:
                        amp_val1 = psi.get_amp(x_i).contract()
                        print(f'Reused Amp val: {amp_val}, Exact Amp val: {amp_val1}, Rel error: {torch.abs(amp_val1 - amp_val) / torch.abs(amp_val1)}')
                        amp_val = amp_val1

                else:
                    if self.env_x_cache is None and self.env_y_cache is None:
                        # check whether we can reuse the cached environment
                        amp = amp_tn.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                        amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                        amp_val = amp.contract() # quimb will address the cached exponent automatically
                    else:
                        config_2d = self.from_1d_to_2d(x_i)
                        # detect the rows that have changed
                        changed_rows, unchanged_rows_above, unchanged_rows_below = self.detect_changed_rows(self.config_ref, x_i)
                        # detect the columns that have changed
                        changed_cols, unchanged_cols_left, unchanged_cols_right = self.detect_changed_cols(self.config_ref, x_i)
                        if len(changed_rows) == 0:
                            key_bot = ('xmax', tuple(torch.cat(tuple(config_2d[self.Lx//2:].to(torch.int))).tolist()))
                            key_top = ('xmin', tuple(torch.cat(tuple(config_2d[:self.Lx//2].to(torch.int))).tolist()))
                            amp_bot = self.env_x_cache[key_bot]
                            amp_top = self.env_x_cache[key_top]
                            amp_val = (amp_bot|amp_top).contract()*10**(self.skeleton.exponent)
                        else:
                            if len(changed_rows) <= len(changed_cols):
                                # for bottom envs, until the last row in the changed rows, we can reuse the env
                                # for top envs, until the first row in the changed rows, we can reuse the env
                                amp_changed_rows = qtn.TensorNetwork([amp_tn.select(amp_tn.x_tag_id.format(row_n)) for row_n in changed_rows])
                                amp_unchanged_bottom_env = qtn.TensorNetwork()
                                amp_unchanged_top_env = qtn.TensorNetwork()
                                if len(unchanged_rows_below) != 0:
                                    amp_unchanged_bottom_env = self.env_x_cache[('xmax', tuple(torch.cat(tuple(config_2d[unchanged_rows_below].to(torch.int))).tolist()))]
                                if len(unchanged_rows_above) != 0:
                                    amp_unchanged_top_env = self.env_x_cache[('xmin', tuple(torch.cat(tuple(config_2d[unchanged_rows_above].to(torch.int))).tolist()))]
                                amp_val = (amp_unchanged_bottom_env|amp_unchanged_top_env|amp_changed_rows).contract() * 10**(self.skeleton.exponent)
                            else:
                                # for left envs, until the first column in the changed columns, we can reuse the env
                                # for right envs, until the last column in the changed columns, we can reuse the env
                                amp_changed_cols = qtn.TensorNetwork([amp_tn.select(amp_tn.y_tag_id.format(col_n)) for col_n in changed_cols])
                                amp_unchanged_left_env = qtn.TensorNetwork()
                                amp_unchanged_right_env = qtn.TensorNetwork()
                                if len(unchanged_cols_left) != 0:
                                    amp_unchanged_left_env = self.env_y_cache[('ymin', tuple(torch.cat(tuple(config_2d[:, unchanged_cols_left].to(torch.int))).tolist()))]
                                if len(unchanged_cols_right) != 0:
                                    amp_unchanged_right_env = self.env_y_cache[('ymax', tuple(torch.cat(tuple(config_2d[:, unchanged_cols_right].to(torch.int))).tolist()))]
                                amp_val = (amp_unchanged_left_env|amp_unchanged_right_env|amp_changed_cols).contract()*10**(self.skeleton.exponent)

            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            
            # if self.debug:
            #     amp_val_exact = psi.get_amp(x_i).contract()
            #     if (amp_val - amp_val_exact).abs() > 1e-4:
            #         print(f'Warning: Reused Amp val and Exact Amp val differ significantly! Reused Amp val: {amp_val}, Exact Amp val: {amp_val_exact}, Rel error: {torch.abs(amp_val_exact - amp_val) / torch.abs(amp_val_exact)}')
                

            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def get_grad(self):
        """"Compute the amplitude gradient by contracting TN without the on-site tensor."""
        if self.debug:
            print('Computing the amplitude gradient by contracting TN without the on-site tensor.')
        self.zero_grad()
        sampled_x = self.config_ref
        amp_tn = self.amp_ref
        index_map = {0: 0, 1: 1, 2: 1, 3: 0}
        array_map = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
            3: torch.tensor([0.0, 1.0])
        }
        config_2d = self.from_1d_to_2d(sampled_x)
        for tid, model_ts_params in self.torch_tn_params.items():
            # utils numbers
            site_2d = self.from_1dsite_to_2dsite(int(tid))

            p_ind = self.skeleton.site_ind_id.format(*site_2d)
            p_ind_order = self.skeleton.tensor_map[int(tid)].inds.index(p_ind)
            on_site_config = int(sampled_x[int(tid)])
            on_site_config_parity = index_map[on_site_config]
            site_tag = self.skeleton.site_tag_id.format(*site_2d)
            input_vec = array_map[on_site_config]

            ts0 = amp_tn.select(site_tag).contract()
            ts_params = ts0.get_params()

            # Reuse cached environment to compute grad_ts
            row_id = site_2d[0]
            # select the cached_env on both sides of the site tensor along the row
            rows_above = list(range(row_id+1, self.Lx))
            rows_below = list(range(0, row_id))
            cached_amp_tn_above = self.env_x_cache[('xmax', tuple(torch.cat(tuple(config_2d[rows_above].to(torch.int))).tolist()))] if rows_above else qtn.TensorNetwork([])
            cached_amp_tn_below = self.env_x_cache[('xmin', tuple(torch.cat(tuple(config_2d[rows_below].to(torch.int))).tolist()))] if rows_below else qtn.TensorNetwork([])
            within_row_sites = list((site_2d[0], col_id) for col_id in range(self.Ly) if col_id != site_2d[1])
            within_row_hole_tn = self.get_local_amp_tensors(within_row_sites, config=sampled_x)
            grad_ts = (within_row_hole_tn | cached_amp_tn_above | cached_amp_tn_below).contract()

            # # Exact contraction of the TN without the site tensor, naive calculation, expensive baseline.
            # ts_left = [ts for ts in amp_tn.tensors if site_tag not in ts.tags]
            # tn_left = qtn.TensorNetwork(ts_left)
            # grad_ts = tn_left.contract()

            grad_ts.data.phase_sync(inplace=True)

            # Back propagate through the final contraction
            ts0.apply_to_arrays(lambda x: x.clone().detach().requires_grad_(True))
            grad_ts.apply_to_arrays(lambda x: x.clone().detach().requires_grad_(False))
            amp_temp = (ts0|grad_ts).contract()*10**(self.skeleton.exponent)
            amp_temp.backward()

            grad_ts_backprop_params = ts0.get_params().copy() # correct gradients
            for blk, ts_temp in ts0.get_params().items():
                grad_ts_backprop_params[blk] = ts_temp.grad

            # select the remaining sectors in model_ts_params
            remaining_sliced_sectors = [sector for sector in sorted(ts_params)]
            remaining_sectors = [sector + (on_site_config_parity,) for sector in sorted(ts_params)]

            for sliced_blk, blk in zip(remaining_sliced_sectors, remaining_sectors):
                data = model_ts_params[str(blk)]
                select_index = torch.argmax(input_vec).item()
                slicer = [slice(None)] * data.ndim
                slicer[p_ind_order] = select_index
                reconstructed_grad_tensor = torch.zeros_like(data)
                reconstructed_grad_tensor[tuple(slicer)] = grad_ts_backprop_params[sliced_blk]
                data.grad = reconstructed_grad_tensor



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
            amp = psi.get_amp(x_i)

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


class PureAttention_Model(wavefunctionModel):
    def __init__(self, phys_dim=4, n_site=None, embedding_dim=32, attention_heads=4, nn_hidden_dim=128, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype

        # Define the neural network
        input_dim = n_site
        self.nn = SelfAttn_block_pos(
            n_site=input_dim,
            num_classes=phys_dim,
            embed_dim=embedding_dim,
            attention_heads=attention_heads,
            dtype=self.param_dtype,
        )

        # for each tensor (labelled by tid), assign a MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim*embedding_dim, nn_hidden_dim),
            nn.Tanh(),
            nn.Linear(nn_hidden_dim, nn_hidden_dim),
            ShiftedSinhYFixed(),
        )
        self.mlp.to(self.param_dtype)

        self.model_structure = {
            'pure attention':
            {'n_site': n_site, 
             'phys_dim': phys_dim, 
             'embedding_dim': embedding_dim, 
             'attention_heads': attention_heads, 
             'nn_hidden_dim': nn_hidden_dim, 
            },
        }
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
            embed_features = self.nn(x_i)
            # Concatenate the embedding features for each site
            embed_features_flat = embed_features.view(-1)  # Flatten the embedding dimensions
            # Compute the amplitude using the MLP
            amp = self.mlp(embed_features_flat)
            batch_amps.append(torch.mean(amp, dim=0))
        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

import torch.nn.functional as F
class PureNN_Model(wavefunctionModel):
    def __init__(self, phys_dim=4, n_site=None, embed_dim=4, nn_hidden_dim=128, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        self.phys_dim = phys_dim
        self.embed_dim = embed_dim

        # Define the neural network
        # self.embed = nn.Embedding(phys_dim//2, embed_dim)
        self.nn = nn.Sequential(
            nn.Linear(2*n_site, nn_hidden_dim),
            nn.Tanh(),
            nn.Linear(nn_hidden_dim, nn_hidden_dim),
            ShiftedSinhYFixed(),
        )
        self.nn.to(self.param_dtype)
        # self.embed.to(self.param_dtype)

        self.model_structure = {
            'pure NN':
            {'n_site': n_site, 
             'phys_dim': phys_dim, 
             'nn_hidden_dim': nn_hidden_dim, 
            },
        }
        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]
    
    def amplitude(self, x):
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            
            netket_xi = torch.tensor(from_quimb_config_to_netket_config(x_i), dtype=self.param_dtype)
            # embedded_xi = self.embed(netket_xi).view(-1)  # Flatten the embedding dimensions
            # embed_features = F.one_hot(x_i.long(), num_classes=self.phys_dim).to(self.param_dtype)
            # embed_features_flat = torch.flatten(embed_features)
            # Compute the amplitude using the MLP
            # amp = self.nn(embed_features_flat)
            # amp = self.nn(embedded_xi)
            # amp = self.nn(x_i)
            amp = self.nn(netket_xi)
            batch_amps.append(torch.mean(amp, dim=0))
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
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
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
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
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
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
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
        There's no positional encoding in the input of the attention block in this model, so the attention output collapse to a simple average of the embeded features.
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
            dtype=self.param_dtype,
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
                nn.LeakyReLU(),
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
            tn_nn_params = reconstruct_tn_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=100, parallel=True)
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
    
class fTN_backflow_attn_Tensorwise_Model_v2(wavefunctionModel):
    """
        For each on-site fermionic tensor with specific shape, assign a narrow on-site projector MLP with corresponding output dimension.
        This is to avoid the large number of parameters in the previous model, where Np = N_neurons * N_TNS.
        Positional encoding is added to the input of the attention block in this model.
        Update from v1: use positional encoding in the attention block and use tanh instead of LeakyReLU in the last MLPs.
    """
    def __init__(self, ftn, max_bond=None, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=torch.float32, eps=5e-3):
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
        
        self.nn = SelfAttn_block_pos(
            n_site=input_dim,
            num_classes=phys_dim,
            embed_dim=embedding_dim,
            attention_heads=attention_heads,
            dtype=self.param_dtype,
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
                nn.Tanh(),
                nn.Linear(nn_final_dim, tn_params_vec.numel()),
            )
            self.mlp[tid].to(self.param_dtype)
        
        # Initialize the MLP last layer weights and biases with small values
        for mlp in self.mlp.values():
            last_layer = mlp[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.uniform_(last_layer.weight, a=-eps, b=eps)
                if last_layer.bias is not None:
                    nn.init.uniform_(last_layer.bias, a=-eps, b=eps)

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
            # tn_nn_params = reconstruct_tn_params(params_vec + (self.nn_eta-10)*nn_correction.detach()+10*nn_correction, params)
            tn_nn_params = reconstruct_tn_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
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
    

class fTN_backflow_attn_Tensorwise_Model_v3(wavefunctionModel):
    """
        For each on-site fermionic tensor with specific shape, assign a narrow on-site projector MLP with corresponding output dimension.
        This is to avoid the large number of parameters in the previous model, where Np = N_neurons * N_TNS.
        Positional encoding is added to the input of the attention block in this model.
        Update from v1: after attention block an position-wise MLP is used to add non-linearity to attention output, and potentially remove layer normalization.
    """
    def __init__(self, ftn, max_bond=None, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=torch.float32, layer_norm=True, position_wise_mlp=True, positional_encoding=True):
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
        
        self.nn = SelfAttn_MLP(
            n_site=input_dim,
            num_classes=phys_dim,
            embed_dim=embedding_dim,
            attention_heads=attention_heads,
            dtype=self.param_dtype,
            layer_norm=layer_norm,
            position_wise_mlp=position_wise_mlp,
            positional_encoding=positional_encoding,
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
                nn.LeakyReLU(),
                nn.Linear(nn_final_dim, tn_params_vec.numel()),
            )
            self.mlp[tid].to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]


        self.model_structure = {
            'fPEPS_backflow_attn_Tensorwise_MLP':
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
            tn_nn_params = reconstruct_tn_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
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


class fTN_backflow_attn_Tensorwise_Model_v4(wavefunctionModel):
    """
        For each on-site fermionic tensor with specific shape, assign a narrow on-site projector MLP with corresponding output dimension.
        This is to avoid the large number of parameters in the previous model, where Np = N_neurons * N_TNS.
        Positional encoding is added to the input of the attention block in this model.
        Update: do not flatten the features from the attention block, instead use them directly in each on-site MLP to generate the tensor correction.
    """
    def __init__(self, ftn, max_bond=None, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, dtype=torch.float32, layer_norm=True, position_wise_mlp=False, positional_encoding=True):
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
        
        self.nn = SelfAttn_MLP(
            n_site=input_dim,
            num_classes=phys_dim,
            embed_dim=embedding_dim,
            attention_heads=attention_heads,
            dtype=self.param_dtype,
            layer_norm=layer_norm,
            position_wise_mlp=position_wise_mlp,
            positional_encoding=positional_encoding,
        )
        # for each tensor (labelled by tid), assign a MLP
        self.mlp = nn.ModuleDict()
        for tid in self.torch_tn_params.keys():
            mlp_input_dim = embedding_dim
            tn_params_dict = {
                tid: params[int(tid)]
            }
            tn_params_vec = flatten_tn_params(tn_params_dict)
            self.mlp[tid] = nn.Sequential(
                nn.Linear(mlp_input_dim, nn_final_dim),
                nn.LeakyReLU(),
                nn.Linear(nn_final_dim, tn_params_vec.numel()),
            )
            self.mlp[tid].to(self.param_dtype)

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]


        self.model_structure = {
            'fPEPS_backflow_attn_Tensorwise_MLP':
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
            nn_features = self.nn(x_i)
            nn_correction = torch.cat([self.mlp[tid](nn_features[int(tid)]) for tid in self.torch_tn_params.keys()])
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_tn_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
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

def get_receptive_field_2d(Lx, Ly, r, site_index_map=lambda i, j, Lx, Ly: i * Ly + j):
    """
        Get the receptive field (OBC) for each site in a square lattice graph.
        Default ordering is zig-zag ordering.
    """
    receptive_field = {}
    for i in range(Lx):
        for j in range(Ly):
            for ix in range(-r+i, r+1+i):
                for jx in range(-r+j, r+1+j):
                    if ix >= 0 and ix < Lx and jx >= 0 and jx < Ly:
                        site_id = site_index_map(i, j, Lx, Ly)
                        if site_id not in receptive_field:
                            receptive_field[site_id] = []
                        receptive_field[site_id].append(site_index_map(ix, jx, Lx, Ly))
    return receptive_field

class fTN_BFA_cluster_Model(wavefunctionModel):
    """
        fPEPS + tensorwise attention backflow NN with finite receptive field
    """
    def __init__(self, fpeps, max_bond=None, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, radius=1, jastrow=False, dtype=torch.float32):
        super().__init__()
        self.param_dtype = dtype
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(fpeps)
        self.skeleton.exponent = 0

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get the receptive field for each site
        self.nn_radius = radius
        self.receptive_field = get_receptive_field_2d(fpeps.Lx, fpeps.Ly, self.nn_radius)

        phys_dim = fpeps.phys_dim()
        # for each tensor (labelled by tid), assign a attention+MLP
        self.nn = nn.ModuleDict()
        for tid in self.torch_tn_params.keys():
            # get the receptive field for the current tensor
            neighbors = self.receptive_field[int(tid)]
            input_dim = len(neighbors)
            on_site_ts_params_dict ={
                tid: params[int(tid)]
            }
            on_site_ts_params_vec = flatten_tn_params(on_site_ts_params_dict)
            self.nn[tid] = SelfAttn_FFNN_block(
                n_site=input_dim,
                num_classes=phys_dim,
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                nn_hidden_dim=nn_final_dim,
                output_dim=on_site_ts_params_vec.numel(),
                dtype=self.param_dtype
            )
        if jastrow:
            global_jastrow_input_dim = fpeps.Lx * fpeps.Ly
            self.jastrow = SelfAttn_FFNN_block(
                    n_site=global_jastrow_input_dim,
                    num_classes=phys_dim,
                    embedding_dim=embedding_dim,
                    attention_heads=attention_heads,
                    nn_hidden_dim=nn_final_dim,
                    output_dim=1,
                    dtype=self.param_dtype
                )
        else:
            self.jastrow = lambda x: torch.zeros(1)

        # Get symmetry
        self.symmetry = fpeps.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS_BFA_cluster':
            {
                'D': fpeps.max_bond(), 
                'Lx': fpeps.Lx, 'Ly': fpeps.Ly, 
                'radius': self.nn_radius,
                'jastrow': jastrow,
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
        
            # For each site get the corresponding input config from x_i and receptive field
            local_configs = {tid: x_i[neighbors] for tid, neighbors in self.receptive_field.items()}
            # Get the corresponding NN output for each site and concatenate
            nn_correction = torch.cat([self.nn[tid](local_configs[int(tid)]) for tid in self.torch_tn_params.keys()])
            # Add the correction to the original parameters
            tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
            # Reconstruct the TN with the new parameters
            psi = qtn.unpack(tn_nn_params, self.skeleton)
            # Get the amplitude
            amp = psi.get_amp(x_i, conj=True)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree) * torch.sum(torch.exp(self.jastrow(x_i)))
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly//2-1])
                amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[psi.Ly//2, psi.Ly-1])
                amp_val = amp.contract() * torch.sum(torch.exp(self.jastrow(x_i)))
            
            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)


class fTN_BFA_cluster_Model_reuse(wavefunctionModel):
    """
        fPEPS + tensorwise attention backflow NN with finite receptive field
    """
    def __init__(self, fpeps, max_bond=None, embedding_dim=32, attention_heads=4, nn_final_dim=4, nn_eta=1.0, radius=1, jastrow=False, dtype=torch.float32, debug=False):
        super().__init__()
        self.param_dtype = dtype
        self.debug = debug
        
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(fpeps)
        self.skeleton.exponent = 0

        # Flatten the dictionary structure and assign each parameter as a part of a ModuleDict
        self.torch_tn_params = nn.ModuleDict({
            str(tid): nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        })

        # Get the receptive field for each site
        self.nn_radius = radius
        self.receptive_field = get_receptive_field_2d(fpeps.Lx, fpeps.Ly, self.nn_radius)

        phys_dim = fpeps.phys_dim()
        # for each tensor (labelled by tid), assign a attention+MLP
        self.nn = nn.ModuleDict()
        for tid in self.torch_tn_params.keys():
            # get the receptive field for the current tensor
            neighbors = self.receptive_field[int(tid)]
            input_dim = len(neighbors)
            on_site_ts_params_dict ={
                tid: params[int(tid)]
            }
            on_site_ts_params_vec = flatten_tn_params(on_site_ts_params_dict)
            self.nn[tid] = SelfAttn_FFNN_block(
                n_site=input_dim,
                num_classes=phys_dim,
                embedding_dim=embedding_dim,
                attention_heads=attention_heads,
                nn_hidden_dim=nn_final_dim,
                output_dim=on_site_ts_params_vec.numel(),
                dtype=self.param_dtype
            )
        if jastrow:
            global_jastrow_input_dim = fpeps.Lx * fpeps.Ly
            self.jastrow = SelfAttn_FFNN_block(
                    n_site=global_jastrow_input_dim,
                    num_classes=phys_dim,
                    embedding_dim=embedding_dim,
                    attention_heads=attention_heads,
                    nn_hidden_dim=nn_final_dim,
                    output_dim=1,
                    dtype=self.param_dtype
                )
        else:
            self.jastrow = lambda x: torch.zeros(1)

        # Get symmetry
        self.symmetry = fpeps.arrays[0].symmetry

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'fPEPS_BFA_cluster':
            {
                'D': fpeps.max_bond(), 
                'Lx': fpeps.Lx, 'Ly': fpeps.Ly, 
                'radius': self.nn_radius,
                'jastrow': jastrow,
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
        self.Lx = fpeps.Lx
        self.Ly = fpeps.Ly
        self._env_x_cache = None
        self._env_y_cache = None
        self.config_ref = None
    
    def from_1d_to_2d(self, config, ordering='snake'):
        if ordering == 'snake':
            config_2d = config.reshape((self.Lx, self.Ly))
        return config_2d
    
    def transform_quimb_env_x_key_to_config_key(self, env_x, config):
        """
            Return a dictionary with the keys being the config rows
        """
        config_2d = self.from_1d_to_2d(config)
        env_x_row_config = {}
        for key in env_x.keys():
            if key[0] == 'xmax': # from bottom to top
                row_n = key[1]
                if row_n != self.Lx-1:
                    rows_config = tuple(torch.cat(tuple(config_2d[row_n+1:].to(torch.int))).tolist())
                    env_x_row_config[('xmax', rows_config)] = env_x[key]
            elif key[0] == 'xmin': # from top to bottom
                row_n = key[1]
                if row_n != 0:
                    rows_config = tuple(torch.cat(tuple(config_2d[:row_n].to(torch.int))).tolist())
                    env_x_row_config[('xmin', rows_config)] = env_x[key]
        return env_x_row_config
    
    def transform_quimb_env_y_key_to_config_key(self, env_y, config):
        """
            Return a dictionary with the keys being the config cols
        """
        config_2d = self.from_1d_to_2d(config)
        env_y_row_config = {}
        for key in env_y.keys():
            if key[0] == 'ymax':
                col_n = key[1]
                if col_n != self.Ly-1:
                    cols_config = tuple(torch.cat(tuple(config_2d[:, col_n+1:].to(torch.int))).tolist())
                    env_y_row_config[('ymax', cols_config)] = env_y[key]
            elif key[0] == 'ymin':
                col_n = key[1]
                if col_n != 0:
                    cols_config = tuple(torch.cat(tuple(config_2d[:, :col_n].to(torch.int))).tolist())
                    env_y_row_config[('ymin', cols_config)] = env_y[key]
        return env_y_row_config

    def cache_env_x(self, amp, config):
        """
            Cache the environment x for the given configuration
        """
        env_x = amp.compute_x_environments(max_bond=self.max_bond, cutoff=0.0)
        env_x_cache = self.transform_quimb_env_x_key_to_config_key(env_x, config)
        self._env_x_cache = env_x_cache
        self.config_ref = config
    
    def cache_env_y(self, amp, config):
        env_y = amp.compute_y_environments(max_bond=self.max_bond, cutoff=0.0)
        env_y_cache = self.transform_quimb_env_y_key_to_config_key(env_y, config)
        self._env_y_cache = env_y_cache
        self.config_ref = config
    
    def cache_env(self, amp, config):
        """
            Cache the environment x and y for the given configuration
        """
        self.cache_env_x(amp, config)
        self.cache_env_y(amp, config)
        
    @property
    def env_x_cache(self):
        """
            Return the cached environment x
        """
        if hasattr(self, '_env_x_cache'):
            return self._env_x_cache
        else:
            return None
        
    @property
    def env_y_cache(self):
        """
            Return the cached environment y
        """
        if hasattr(self, '_env_y_cache'):
            return self._env_y_cache
        else:
            return None
    
    def clear_env_x_cache(self):
        """
            Clear the cached environment x
        """
        self._env_x_cache = None

    def clear_env_y_cache(self):
        """
            Clear the cached environment y
        """
        self._env_y_cache = None
    
    def clear_wavefunction_env_cache(self):
        self.clear_env_x_cache()
        self.clear_env_y_cache()
        self.config_ref = None
    
    def detect_effected_rows(self, config_ref, new_config):
        """
            Detect the rows that have been effected in the new configuration
        """
        config_ref_2d = self.from_1d_to_2d(config_ref)
        new_config_2d = self.from_1d_to_2d(new_config)
        effected_rows = set()
        for i in range(self.Lx):
            if not torch.equal(config_ref_2d[i], new_config_2d[i]):
                effected_row_top = i-self.nn_radius if i-self.nn_radius >= 0 else 0
                effected_row_bottom = i+self.nn_radius if i+self.nn_radius < self.Lx else self.Lx-1
                effected_rows.update(list(range(effected_row_top, effected_row_bottom+1)))
        effected_rows = sorted(effected_rows)

        if len(effected_rows) == 0:
            return [], [], []
        uneffected_rows_above = list(range(effected_rows[0]))
        uneffected_rows_below = list(range(effected_rows[-1]+1, self.Lx))
        return effected_rows, uneffected_rows_above, uneffected_rows_below
    
    def detect_effected_cols(self, config_ref, new_config):
        """
            Detect the cols that have been effected in the new configuration
        """
        config_ref_2d = self.from_1d_to_2d(config_ref)
        new_config_2d = self.from_1d_to_2d(new_config)
        effected_cols = set()
        for j in range(self.Ly):
            if not torch.equal(config_ref_2d[:, j], new_config_2d[:, j]):
                effected_col_left = j-self.nn_radius if j-self.nn_radius >= 0 else 0
                effected_col_right = j+self.nn_radius if j+self.nn_radius < self.Ly else self.Ly-1
                effected_cols.update(list(range(effected_col_left, effected_col_right+1)))
        effected_cols = sorted(effected_cols)

        if len(effected_cols) == 0:
            return [], [], []
        uneffected_cols_left = list(range(effected_cols[0]))
        uneffected_cols_right = list(range(effected_cols[-1]+1, self.Ly))
        return effected_cols, uneffected_cols_left, uneffected_cols_right
    
    def detect_changed_rows(self, config_ref, new_config):
        """
            Detect the rows that have changed in the new configuration
        """
        config_ref_2d = self.from_1d_to_2d(config_ref)
        new_config_2d = self.from_1d_to_2d(new_config)
        changed_rows = []
        for i in range(self.Lx):
            if not torch.equal(config_ref_2d[i], new_config_2d[i]):
                changed_rows.append(i)
        if len(changed_rows) == 0:
            return [], [], []
        unchanged_rows_above = list(range(changed_rows[0]))
        unchanged_rows_below = list(range(changed_rows[-1]+1, self.Lx))
        return changed_rows, unchanged_rows_above, unchanged_rows_below
    
    def detect_changed_cols(self, config_ref, new_config):
        """
            Detect the columns that have changed in the new configuration
        """
        config_ref_2d = self.from_1d_to_2d(config_ref)
        new_config_2d = self.from_1d_to_2d(new_config)
        changed_cols = []
        for i in range(self.Ly):
            if not torch.equal(config_ref_2d[:, i], new_config_2d[:, i]):
                changed_cols.append(i)
        if len(changed_cols) == 0:
            return [], [], []
        unchanged_cols_left = list(range(changed_cols[0]))
        unchanged_cols_right = list(range(changed_cols[-1]+1, self.Ly))
        return changed_cols, unchanged_cols_left, unchanged_cols_right
    
    def update_env_x_cache(self, config):
        """
            Update the cached environment x for the given configuration
        """
        if self.env_x_cache is not None:
            self.clear_env_x_cache()
        amp_tn = self.get_amp_tn(config)
        self.cache_env_x(amp_tn, config)
        self.config_ref = config
    
    def update_env_x_cache_to_row(self, config, row_id, from_which='xmin'):
        config_2d = self.from_1d_to_2d(config)
        amp_tn = self.get_amp_tn(config)
        upper_effected_row_id = row_id - self.nn_radius if row_id - self.nn_radius >= 0 else 0
        lower_effected_row_id = row_id + self.nn_radius if row_id + self.nn_radius < self.Lx else self.Lx-1
        row_id = upper_effected_row_id if from_which == 'xmin' else lower_effected_row_id
        # select the row_tn in the amp_tn that corresponds to the row_id
        row_tn = amp_tn.select(amp_tn.x_tag_id.format(row_id))
        if from_which == 'xmin':
            # check the whether previous env_x cache already contains the env w.r.t. row with id upper_effected_row_id
            key_prev_rows = ('xmin', tuple(torch.cat(tuple(config_2d[:row_id].to(torch.int))).tolist())) if row_id != 0 else ()
            new_env_key = ('xmin', tuple(torch.cat(tuple(config_2d[:row_id+1].to(torch.int))).tolist()))
            xrange = (0, row_id)
        elif from_which == 'xmax':
            # check the whether previous env_x cache already contains the env w.r.t. row with id lower_effected_row_id
            key_prev_rows = ('xmax', tuple(torch.cat(tuple(config_2d[row_id+1:].to(torch.int))).tolist())) if row_id != self.Lx-1 else ()
            new_env_key = ('xmax', tuple(torch.cat(tuple(config_2d[row_id:].to(torch.int))).tolist()))
            xrange = (row_id, self.Lx-1)
        else:
            raise ValueError("from_which must be either 'xmin' or 'xmax'")

        if self.env_x_cache is not None: 
            if key_prev_rows in self.env_x_cache:
                prev_env_x = self.env_x_cache[key_prev_rows]
                new_env_tn = prev_env_x | row_tn
                new_env_tn.contract_boundary_from_(max_bond=self.max_bond, cutoff=0.0, xrange=xrange, from_which=from_which, yrange=(0, self.Ly-1))
                new_env_x_cache = {new_env_key: new_env_tn}
            else:
                new_env_x = amp_tn.compute_environments(max_bond=self.max_bond, cutoff=0.0, xrange=(0, row_id+1) if from_which=='xmin' else (row_id-1, self.Lx-1), from_which=from_which)
                new_env_x_cache = self.transform_quimb_env_x_key_to_config_key(new_env_x, config)
        else:
            new_env_x = amp_tn.compute_environments(max_bond=self.max_bond, cutoff=0.0, xrange=(0, row_id+1) if from_which=='xmin' else (row_id-1, self.Lx-1), from_which=from_which)
            new_env_x_cache = self.transform_quimb_env_x_key_to_config_key(new_env_x, config)
        # add the new env_x to the cache
        if self.env_x_cache is None:
            self._env_x_cache = new_env_x_cache
        else:
            self._env_x_cache.update(new_env_x_cache)
        self.config_ref = config
    
    def update_env_y_cache(self, config):
        """
            Update the cached environment y for the given configuration
        """
        if self.env_y_cache is not None:
            self.clear_env_y_cache()
        amp_tn = self.get_amp_tn(config)
        self.cache_env_y(amp_tn, config)
        self.config_ref = config
    
    def update_env_y_cache_to_col(self, config, col_id, from_which='ymin'):
        config_2d = self.from_1d_to_2d(config)
        amp_tn = self.get_amp_tn(config)
        left_effected_col_id = col_id - self.nn_radius if col_id - self.nn_radius >= 0 else 0
        right_effected_col_id = col_id + self.nn_radius if col_id + self.nn_radius < self.Ly else self.Ly-1
        col_id = left_effected_col_id if from_which == 'ymin' else right_effected_col_id
        # select the col_tn in the amp_tn that corresponds to the col_id
        col_tn = amp_tn.select(amp_tn.y_tag_id.format(col_id))
        # check the whether previous env_y cache already contains the env w.r.t. col with id col_id
        if from_which == 'ymin':
            key_prev_cols = ('ymin', tuple(torch.cat(tuple(config_2d[:, :col_id].to(torch.int))).tolist())) if col_id != 0 else ()
            new_env_key = ('ymin', tuple(torch.cat(tuple(config_2d[:, :col_id+1].to(torch.int))).tolist()))
            yrange = (0, col_id)
        elif from_which == 'ymax':
            key_prev_cols = ('ymax', tuple(torch.cat(tuple(config_2d[:, col_id+1:].to(torch.int))).tolist())) if col_id != self.Ly-1 else ()
            new_env_key = ('ymax', tuple(torch.cat(tuple(config_2d[:, col_id:].to(torch.int))).tolist()))
            yrange = (col_id, self.Ly-1)
        else:
            raise ValueError("from_which must be either 'ymin' or 'ymax'")
        if self.env_y_cache is not None:
            if key_prev_cols in self.env_y_cache:
                prev_env_y = self.env_y_cache[key_prev_cols]
                new_env_tn = prev_env_y | col_tn
                new_env_tn.contract_boundary_from_(max_bond=self.max_bond, cutoff=0.0, yrange=yrange, from_which=from_which, xrange=(0, self.Lx-1))
                new_env_y_cache = {new_env_key: new_env_tn}
            else:
                new_env_y = amp_tn.compute_environments(max_bond=self.max_bond, cutoff=0.0, yrange=(0, col_id+1) if from_which=='ymin' else (col_id-1, self.Ly-1), from_which=from_which)
                new_env_y_cache = self.transform_quimb_env_y_key_to_config_key(new_env_y, config)
        else:
            new_env_y = amp_tn.compute_environments(max_bond=self.max_bond, cutoff=0.0, yrange=(0, col_id+1) if from_which=='ymin' else (col_id-1, self.Ly-1), from_which=from_which)
            new_env_y_cache = self.transform_quimb_env_y_key_to_config_key(new_env_y, config)
        # add the new env_y to the cache
        if self.env_y_cache is None:
            self._env_y_cache = new_env_y_cache
        else:
            self._env_y_cache.update(new_env_y_cache)
        self.config_ref = config
    
    def get_amp_tn(self, config):
        """
            Get the amplitude tensor network for the given configuration
        """
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            int(tid): {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_tn_params.items()
        }
        params_vec = flatten_tn_params(params)

        # Get the NN correction to the parameters, concatenate the results for each tensor
        nn_correction = torch.cat([self.nn[str(tid)](config[neighbors]) for tid, neighbors in self.receptive_field.items()])
        # Add the correction to the original parameters
        tn_nn_params = reconstruct_proj_params(params_vec + self.nn_eta*nn_correction, params)
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(tn_nn_params, self.skeleton)
        
        # Get the amplitude
        amp = psi.get_amp(config, conj=True)

        return amp
    
    def amplitude(self, x):
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
            amp = self.get_amp_tn(x_i)

            if self.max_bond is None:
                amp = amp
                if self.tree is None:
                    opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
                    self.tree = amp.contraction_tree(optimize=opt)
                amp_val = amp.contract(optimize=self.tree) * torch.sum(torch.exp(self.jastrow(x_i)))
            else:
                if self.cache_env_mode:
                    self.cache_env_x(amp, x_i)
                    # self.cache_env_y(amp, x_i)
                    assert (self.config_ref == x_i).all()

                    config_2d = self.from_1d_to_2d(x_i)
                    key_bot = ('xmax', tuple(torch.cat(tuple(config_2d[self.Lx//2:].to(torch.int))).tolist()))
                    key_top = ('xmin', tuple(torch.cat(tuple(config_2d[:self.Lx//2].to(torch.int))).tolist()))
                    amp_bot = self.env_x_cache[key_bot]
                    amp_top = self.env_x_cache[key_top]
                    amp_val = (amp_bot|amp_top).contract() * torch.sum(torch.exp(self.jastrow(x_i)))
                else:
                    if self.env_x_cache is None and self.env_y_cache is None:
                        # check whether we can reuse the cached environment
                        amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, self.Ly//2-1])
                        amp = amp.contract_boundary_from_ymax(max_bond=self.max_bond, cutoff=0.0, yrange=[self.Ly//2, self.Ly-1])
                        amp_val = amp.contract() * torch.sum(torch.exp(self.jastrow(x_i)))
                    else:
                        config_2d = self.from_1d_to_2d(x_i)
                        # detect the rows that have been effected in the new configuration
                        effected_rows, uneffected_rows_above, uneffected_rows_below = self.detect_effected_rows(self.config_ref, x_i)
                        
                        # detect the cols that have been effected in the new configuration
                        effected_cols, uneffected_cols_left, uneffected_cols_right = self.detect_effected_cols(self.config_ref, x_i)

                        changed_rows, _, _ = self.detect_changed_rows(self.config_ref, x_i)
                        changed_cols, _, _ = self.detect_changed_cols(self.config_ref, x_i)
                        if len(changed_rows) == 0:
                            key_bot = ('xmax', tuple(torch.cat(tuple(config_2d[self.Lx//2:].to(torch.int))).tolist()))
                            key_top = ('xmin', tuple(torch.cat(tuple(config_2d[:self.Lx//2].to(torch.int))).tolist()))
                            amp_bot = self.env_x_cache[key_bot]
                            amp_top = self.env_x_cache[key_top]
                            amp_val = (amp_bot|amp_top).contract() * torch.sum(torch.exp(self.jastrow(x_i)))
                        else:
                            if len(changed_rows) <= len(changed_cols):
                                # for bottom envs, until the last effected row, we can reuse the bottom envs
                                # for top envs, until the first effected row, we can reuse the top envs
                                row_tag_list = [amp.x_tag_id.format(row_n) for row_n in effected_rows]
                                amp_effected_rows = amp.select(row_tag_list, which='any')

                                amp_uneffected_bottom_env = qtn.TensorNetwork2D()
                                amp_uneffected_bottom_env._site_tag_id = amp._site_tag_id
                                amp_uneffected_bottom_env._x_tag_id = amp._x_tag_id
                                amp_uneffected_bottom_env._y_tag_id = amp._y_tag_id
                                amp_uneffected_bottom_env._Lx = self.Lx
                                amp_uneffected_bottom_env._Ly = self.Ly
                                amp_uneffected_top_env = qtn.TensorNetwork2D()
                                amp_uneffected_top_env._site_tag_id = amp._site_tag_id
                                amp_uneffected_top_env._x_tag_id = amp._x_tag_id
                                amp_uneffected_top_env._y_tag_id = amp._y_tag_id
                                amp_uneffected_top_env._Lx = self.Lx
                                amp_uneffected_top_env._Ly = self.Ly


                                if len(uneffected_rows_below) != 0:
                                    amp_uneffected_bottom_env = self.env_x_cache[('xmax', tuple(torch.cat(tuple(config_2d[uneffected_rows_below].to(torch.int))).tolist()))]
                                if len(uneffected_rows_above) != 0:
                                    amp_uneffected_top_env = self.env_x_cache[('xmin', tuple(torch.cat(tuple(config_2d[uneffected_rows_above].to(torch.int))).tolist()))]
                                amp_val_tn = amp_effected_rows|amp_uneffected_bottom_env|amp_uneffected_top_env

                                middle_row = effected_rows[0] + (effected_rows[-1] - effected_rows[0])//2
                                if len(uneffected_rows_above) <= len(uneffected_rows_below):
                                    amp_val_tn.contract_boundary_from_xmin_(max_bond=self.max_bond, cutoff=0.0, xrange=[0, middle_row])
                                    amp_val_tn.contract_boundary_from_xmax_(max_bond=self.max_bond, cutoff=0.0, xrange=[middle_row+1, self.Lx-1])
                                else:
                                    amp_val_tn.contract_boundary_from_xmax_(max_bond=self.max_bond, cutoff=0.0, xrange=[0, middle_row-1])
                                    amp_val_tn.contract_boundary_from_xmin_(max_bond=self.max_bond, cutoff=0.0, xrange=[middle_row, self.Lx-1])

                                amp_val = amp_val_tn.contract() * torch.sum(torch.exp(self.jastrow(x_i)))
                            else:
                                col_tag_list = [amp.y_tag_id.format(col_n) for col_n in effected_cols]
                                amp_effected_cols = amp.select(col_tag_list, which='any')

                                amp_uneffected_left_env = qtn.TensorNetwork2D()
                                amp_uneffected_left_env._site_tag_id = amp._site_tag_id
                                amp_uneffected_left_env._x_tag_id = amp._x_tag_id
                                amp_uneffected_left_env._y_tag_id = amp._y_tag_id
                                amp_uneffected_left_env._Lx = self.Lx
                                amp_uneffected_left_env._Ly = self.Ly
                                amp_uneffected_right_env = qtn.TensorNetwork2D()
                                amp_uneffected_right_env._site_tag_id = amp._site_tag_id
                                amp_uneffected_right_env._x_tag_id = amp._x_tag_id
                                amp_uneffected_right_env._y_tag_id = amp._y_tag_id
                                amp_uneffected_right_env._Lx = self.Lx
                                amp_uneffected_right_env._Ly = self.Ly

                                if len(uneffected_cols_left) != 0:
                                    amp_uneffected_left_env = self.env_y_cache[('ymin', tuple(torch.cat(tuple(config_2d[:, uneffected_cols_left].to(torch.int))).tolist()))]
                                if len(uneffected_cols_right) != 0:
                                    amp_uneffected_right_env = self.env_y_cache[('ymax', tuple(torch.cat(tuple(config_2d[:, uneffected_cols_right].to(torch.int))).tolist()))]
                                amp_val_tn = amp_effected_cols|amp_uneffected_left_env|amp_uneffected_right_env

                                middle_col = effected_cols[0] + (effected_cols[-1] - effected_cols[0])//2
                                if len(uneffected_cols_left) <= len(uneffected_cols_right):
                                    amp_val_tn.contract_boundary_from_ymin_(max_bond=self.max_bond, cutoff=0.0, yrange=[0, middle_col])
                                    amp_val_tn.contract_boundary_from_ymax_(max_bond=self.max_bond, cutoff=0.0, yrange=[middle_col+1, self.Ly-1])
                                else:
                                    amp_val_tn.contract_boundary_from_ymax_(max_bond=self.max_bond, cutoff=0.0, yrange=[0, middle_col-1])
                                    amp_val_tn.contract_boundary_from_ymin_(max_bond=self.max_bond, cutoff=0.0, yrange=[middle_col, self.Ly-1])

                                amp_val = amp_val_tn.contract() * torch.sum(torch.exp(self.jastrow(x_i)))

            if amp_val==0.0:
                amp_val = torch.tensor(0.0)
            
            if self.debug:
                print(f"Reused Amp/Exact Amp: {amp_val/(self.get_amp_tn(x_i).contract()* torch.sum(torch.exp(self.jastrow(x_i))))}")
                
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
            nn.LeakyReLU(),
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
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
            nn.LeakyReLU(),
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
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
            nn.LeakyReLU(),
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
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

    def forward(self, x, **kwargs):
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
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
        return torch.cat([param.data.reshape(-1) for param in self.parameters()])
    
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
        param_grad_vec = torch.cat([param.grad.reshape(-1) if param.grad is not None else torch.zeros_like(param).reshape(-1) for param in self.parameters()])
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
    
    def forward(self, x, **kwargs):
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
            kernel_init(torch.empty(self.hilbert.size, self.hilbert.n_fermions, dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.size, self.hilbert.n_fermions, dtype=self.param_dtype)
        )

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'SlaterDeterminant':{'N_orbitals': self.hilbert.size, 'N_fermions': self.hilbert.n_fermions}
        }
    
    def _determinant(self, A):
        # Compute the determinant of matrix A
        det = torch.linalg.det(A)
        return det

    def forward(self, x):
        # Define the slater determinant function manually to loop over inputs
        def slater_det(n):
            # Find the positions of the occupied orbitals
            R = torch.nonzero(n, as_tuple=False).squeeze()
            # Extract the Nf x Nf submatrix of M corresponding to the occupied orbitals
            A = self.M[R]
            return self._determinant(A)
        
        # Apply slater_det to each element in the batch
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=torch.int)
            n_i = torch.Tensor(from_quimb_config_to_netket_config(x_i), dtype=torch.int)
            amp_val=slater_det(n_i)
            batch_amps.append(amp_val)
        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)

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

from vmc_torch.fermion_utils import from_quimb_config_to_netket_config, from_netket_config_to_quimb_config

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
    def __init__(self, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64, nn_eta=1):
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
            nn.Linear(self.hilbert.size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.hilbert.size*self.hilbert.n_fermions),
            ShiftedSinhYFixed(),
        )

        # Convert NNs to the appropriate data type
        self.nn.to(self.param_dtype)

        self.nn_eta = nn_eta

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'Neuralbackflow':{'N_site': self.hilbert.size, 'N_fermions': self.hilbert.n_fermions, 'N_fermions_per_spin': self.hilbert.n_fermions_per_spin, 'nn_eta': self.nn_eta}
        }

    def amplitude(self, x):
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        # Define the slater determinant function manually to loop over inputs
        def backflow_det(n):
            # Compute the backflow matrix F using the neural network
            F = self.nn(n)
            if self.nn_eta != 0:
                M  = self.M + F.reshape(self.M.shape)*self.nn_eta
            elif self.nn_eta == 0: # Pure Slater determinant calculation
                M  = self.M
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

class NNBF_attention(wavefunctionModel):
    """Assuming total Sz=0."""
    def __init__(self, nsite, hilbert, kernel_init=None, param_dtype=torch.float32, hidden_dim=64, nn_eta=1, embed_dim=16, attention_heads=4, phys_dim=4, spinflip_symmetry=False):
        super(NNBF_attention, self).__init__()
        
        self.hilbert = hilbert
        self.param_dtype = param_dtype
        self.spinflip_symmetry = spinflip_symmetry
        
        # Initialize the parameter M (N x Nf matrix)
        self.M = nn.Parameter(
            kernel_init(torch.empty(self.hilbert.size, self.hilbert.n_fermions, dtype=self.param_dtype)) 
            if kernel_init is not None 
            else torch.randn(self.hilbert.size, self.hilbert.n_fermions, dtype=self.param_dtype)
        )
        if self.spinflip_symmetry:
            self.M_flip = nn.Parameter(
                kernel_init(torch.empty(self.hilbert.size, self.hilbert.size - self.hilbert.n_fermions, dtype=self.param_dtype)) 
                if kernel_init is not None 
                else torch.randn(self.hilbert.size, self.hilbert.size-self.hilbert.n_fermions, dtype=self.param_dtype)
            )
        
        self.attn = SelfAttn_block_pos(
            nsite,
            num_classes=phys_dim,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            dtype=self.param_dtype,
        )
        
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.phys_dim = phys_dim
        self.nsite = nsite

         # Initialize the neural network layer, input is n and output a matrix with the same shape as M
        self.nn = nn.Sequential(
            nn.Linear(embed_dim*nsite, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.hilbert.size*self.hilbert.n_fermions),
        )
        if self.spinflip_symmetry:
            self.nn_flip = nn.Sequential(
                nn.Linear(embed_dim*nsite, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.hilbert.size*self.hilbert.n_fermions),
            )

        # Convert NNs to the appropriate data type
        self.nn.to(self.param_dtype)
        if self.spinflip_symmetry:
            self.nn_flip.to(self.param_dtype)
        self.nn_eta = nn_eta

        # Store the shapes of the parameters
        self.param_shapes = [param.shape for param in self.parameters()]

        self.model_structure = {
            'NNBF w attention':{
                'N_site': self.nsite, 
                'N_fermions': self.hilbert.n_fermions, 
                'N_fermions_per_spin': self.hilbert.n_fermions_per_spin, 
                'nn_eta': self.nn_eta,
                'embed_dim': self.embed_dim,
                'attention_heads': self.attention_heads,
                'phys_dim': self.phys_dim,
                'spinflip_symmetry': self.spinflip_symmetry,
                }
        }

    def amplitude(self, x):
        # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        # Define the slater determinant function manually to loop over inputs
        def backflow_det(x, spinflip_symmetry=False):
            n = torch.tensor(from_quimb_config_to_netket_config(x), dtype=self.param_dtype)
            if spinflip_symmetry:
                n = torch.abs(1-n) # 0->1, 1->0
                x = from_netket_config_to_quimb_config(n)
                x = torch.tensor(x, dtype=torch.int)

            if self.nn_eta != 0:
                # Compute the attention output
                attn_features = self.attn(x).view(-1)
                # Compute the backflow matrix F using the neural network
                F = self.nn(attn_features) if not spinflip_symmetry else self.nn_flip(attn_features)
                M  = self.M + F.reshape(self.M.shape)*self.nn_eta if not spinflip_symmetry else self.M_flip + F.reshape(self.M_flip.shape)*self.nn_eta
            elif self.nn_eta == 0: # Pure Slater determinant calculation
                M  = self.M if not spinflip_symmetry else self.M_flip

            # Find the positions of the occupied orbitals
            R = torch.nonzero(n, as_tuple=False).squeeze()

            # Extract the 2Nf x Nf submatrix of M corresponding to the occupied orbitals
            A = M[R]
            det = torch.linalg.det(A)
            amp = det

            return amp
        
        batch_amps = []
        for x_i in x:
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=torch.int)

            if self.spinflip_symmetry:
                amp_val  = (backflow_det(x_i, spinflip_symmetry=True) + backflow_det(x_i, spinflip_symmetry=False)) / 2
            else:
                amp_val = backflow_det(x_i, spinflip_symmetry=False) 
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
