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

from vmc_torch.fermion_utils import insert_proj_peps, flatten_proj_params, reconstruct_proj_params
from vmc_torch.global_var import DEBUG, set_debug

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

class fTNModel(torch.nn.Module):

    def __init__(self, ftn, max_bond=None):
        super().__init__()
        
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
        self.max_bond = max_bond
        
    
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
    
    def from_vec_to_params(self, vec, quimb_format=False):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        # XXX: useful at all?
        params = {}
        idx = 0
        for tid, blk_array in self.torch_tn_params.items():
            params[tid] = {}
            for sector, data in blk_array.items():
                shape = data.shape
                size = data.numel()
                if quimb_format:
                    params[tid][ast.literal_eval(sector)] = vec[idx:idx+size].view(shape)
                else:
                    params[tid][sector] = vec[idx:idx+size].view(shape)
                idx += size
        return params
    
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
            if self.max_bond is None:
                batch_amps.append(amp.contract())
            else:
                amp = amp.contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly-2])
                batch_amps.append(amp.contract())

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)


class PEPS_model(torch.nn.Module):
    def __init__(self, peps, max_bond=None):
        super().__init__()
        self.peps = peps
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
                return peps.isel({peps.site_inds[i]: int(s) for i, s in enumerate(xi)}).contract_boundary_from_ymin(max_bond=self.max_bond, cutoff=0.0, yrange=[0, peps.Ly-2]).contract()
        
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


class fTN_NNiso_Model(torch.nn.Module):
    
    def __init__(self, ftn, max_bond, nn_hidden_dim=64, nn_eta=1e-3, param_dtype=torch.float32):
        super().__init__()
        self.max_bond = max_bond
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
        dummy_proj_params_vec = flatten_proj_params(dummy_proj_params)
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

    def product_bra_state(self, config, peps, symmetry='Z2'):
        """Spinless fermion product bra state."""
        product_tn = qtn.TensorNetwork()
        backend = peps.tensors[0].data.backend
        iterable_oddpos = iter(range(2*peps.nsites+1))
        for n, site in zip(config, peps.sites):
            p_ind = peps.site_ind_id.format(*site)
            p_tag = peps.site_tag_id.format(*site)
            tid = peps.sites.index(site)
            nsites = peps.nsites
            # use autoray to ensure the correct backend is used
            with ar.backend_like(backend):
                if symmetry == 'Z2':
                    data = [sr.Z2FermionicArray.from_blocks(blocks={(0,):do('array', [1.0,], like=backend)}, duals=(True,),symmetry='Z2', charge=0, oddpos=2*tid+1), # It doesn't matter if oddpos is None for even parity tensor.
                            sr.Z2FermionicArray.from_blocks(blocks={(1,):do('array', [1.0,], like=backend)}, duals=(True,),symmetry='Z2',charge=1, oddpos=2*tid+1)
                        ]
                elif symmetry == 'U1':
                    data = [sr.U1FermionicArray.from_blocks(blocks={(0,):do('array', [1.0,], like=backend)}, duals=(True,),symmetry='U1', charge=0, oddpos=2*tid+1),
                            sr.U1FermionicArray.from_blocks(blocks={(1,):do('array', [1.0,], like=backend)}, duals=(True,),symmetry='U1', charge=1, oddpos=2*tid+1)
                        ]
            tsr_data = data[int(n)] # BUG: does not fit in jax compilation, a concrete value is needed for traced arrays
            tsr = qtn.Tensor(data=tsr_data, inds=(p_ind,),tags=(p_tag, 'bra'))
            product_tn |= tsr
        return product_tn

    def get_amp(self, peps, config, inplace=False, symmetry='Z2', conj=True):
        """Get the amplitude of a configuration in a PEPS."""
        if not inplace:
            peps = peps.copy()
        if conj:
            amp = peps|self.product_bra_state(config, peps, symmetry).conj()
        else:
            amp = peps|self.product_bra_state(config, peps, symmetry)
        for site in peps.sites:
            site_tag = peps.site_tag_id.format(*site)
            amp.contract_(tags=site_tag)

        amp.view_as_(
            qtn.PEPS,
            site_ind_id="k{},{}",
            site_tag_id="I{},{}",
            x_tag_id="X{}",
            y_tag_id="Y{}",
            Lx=peps.Lx,
            Ly=peps.Ly,
        )
        return amp
        
    
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
            amp = self.get_amp(psi, x_i, symmetry=self.symmetry, conj=True)

            # Insert projectors
            amp_w_proj = insert_proj_peps(amp, max_bond=self.max_bond, yrange=[0, psi.Ly-2])
            amp_tn, proj_tn = amp_w_proj.partition(tags='proj')
            proj_params, proj_skeleton = qtn.pack(proj_tn)
            proj_params_vec = flatten_proj_params(proj_params)

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

            # contract column by column
            
            # batch_amps.append(torch.tensor(new_amp_w_proj.contract(), dtype=torch.float32, requires_grad=True))
            batch_amps.append(new_amp_w_proj.contract())

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)


class fTN_NN_Model(torch.nn.Module):
    
    def __init__(self, ftn, max_bond, nn_hidden_dim=64, nn_eta=1e-3, param_dtype=torch.float32):
        super().__init__()
        self.max_bond = max_bond
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
        dummy_amp = ftn.get_amp(dummy_config, inplace=False, conj=True)
        dummy_amp_2row = dummy_amp.contract_boundary_from_ymin(max_bond=max_bond, cutoff=0.0, yrange=[0, ftn.Ly//2-1])
        dummy_amp_2row = dummy_amp_2row.contract_boundary_from_ymax(max_bond=max_bond, cutoff=0.0, yrange=[ftn.Ly//2, ftn.Ly-1])
        dummy_2row_params, dummy_2row_skeleton = qtn.pack(dummy_amp_2row)
        dummy_2row_params_vec = flatten_proj_params(dummy_2row_params)
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
            assert self.N_fermion %2 == sum(self.parity_config) % 2, "The number of fermions must match the parity of the Z2-TNS."

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
            amp_2row_params_vec = flatten_proj_params(amp_2row_params)
            
            # Check x_i type
            if not type(x_i) == torch.Tensor:
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
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
    
# Transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layer for integer input sequence
        self.embedding = nn.Embedding(input_size, d_model)
        
        # Positional encoding for fixed-length sequences
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        
        # Linear layer for output generation (predicting floating-point values)
        self.fc_out = nn.Linear(d_model, output_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Encode source (input) sequence
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        src = self.pos_encoder(src)
        
        # Positional encoding for target (output) sequence
        tgt = self.pos_encoder(tgt)
        
        # Apply transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Predict floating-point values for each output token
        output = self.fc_out(output)

        # Transpose output to match the shape [batch_size, seq_len, output_size]
        output = output.transpose(0, 1)
        
        return output
    

class fTN_Transformer_Model(torch.nn.Module):
    
    def __init__(self, ftn, max_bond, nn_eta=1e-3, d_model=128, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.max_bond = max_bond
        self.nn_eta = nn_eta
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
        self.d_model = d_model
        self.transformer = TransformerModel(
            input_size=ftn.nsites, 
            output_size=1, 
            d_model=self.d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
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
            amp_2row_params_vec = flatten_proj_params(amp_2row_params)
            vec_len = len(amp_2row_params_vec)
            
            # Check x_i type
            if not type(x_i) == torch.Tensor or x_i.dtype != torch.int32:
                x_i = torch.tensor(x_i, dtype=torch.int32)

            # Input of the transformer
            src = x_i.unsqueeze(0) # Shape: [batch_size==1, seq_len]
            # Target output of the transformer
            tgt = torch.zeros((1, vec_len, self.d_model))
            # Forward pass
            nn_output = self.transformer(src, tgt)
            # concatenate the output to get the final vector of length vec_len
            nn_output = nn_output.view(-1)
            # Add NN output
            amp_2row_params_vec = amp_2row_params_vec + self.nn_eta*nn_output
            # Reconstruct the proj parameters
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
