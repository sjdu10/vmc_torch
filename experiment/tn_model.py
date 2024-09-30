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
    
    def params_grad_to_vec(self):
        param_grad_vec = torch.cat([param.grad.flatten() if param.grad is not None else torch.zeros_like(param).flatten() for param in self.parameters()])
        return param_grad_vec

    def clear_grad(self):
        for param in self.parameters():
            param.grad = None
    
    def from_vec_to_params(self, vec, quimb_format=False):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {}
        idx = 0
        for tid, blk_array in self.torch_params.items():
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
            amp = self.get_amp(psi, x_i, symmetry=self.symmetry, conj=True)
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


class fTN_NNiso_Model(torch.nn.Module):
    
    def __init__(self, ftn, max_bond, nn_hidden_dim=64, nn_eta=1e-3):
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
                x_i = torch.tensor(x_i, dtype=torch.float32)
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