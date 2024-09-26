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

from fermion_utils import insert_proj_peps, flatten_proj_params, reconstruct_proj_params
from global_var import DEBUG, set_debug

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

class fTNModel(torch.nn.Module):

    def __init__(self, ftn):
        super().__init__()
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(ftn)

        # Flatten the dictionary structure and assign each parameter
        self.torch_params = {
            tid: nn.ParameterDict({
                str(sector): nn.Parameter(data)
                for sector, data in blk_array.items()
            })
            for tid, blk_array in params.items()
        }

        # Get symmetry
        self.symmetry = ftn.arrays[0].symmetry

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
        
    def parameters(self):
        # Manually yield all parameters from the nested structure
        for tid_dict in self.torch_params.values():
            for param in tid_dict.values():
                yield param
    
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
        if isinstance(new_params, torch.Tensor):
            new_params = self.from_vec_to_params(new_params)
        # Update the parameters manually
        with torch.no_grad():
            for tid, blk_array in new_params.items():
                for sector, data in blk_array.items():
                    self.torch_params[tid][sector].data = data

    
    def amplitude(self, x):
        # Reconstruct the original parameter structure (by unpacking from the flattened dict)
        params = {
            tid: {
                ast.literal_eval(sector): data
                for sector, data in blk_array.items()
            }
            for tid, blk_array in self.torch_params.items()
        }
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
       # `x` is expected to be batched as (batch_size, input_dim)
        # Loop through the batch and compute amplitude for each sample
        batch_amps = []
        for x_i in x:
            amp = self.get_amp(psi, x_i, symmetry=self.symmetry, conj=True)
            batch_amps.append(amp.contract())

        # Return the batch of amplitudes stacked as a tensor
        return torch.stack(batch_amps)
    
    def forward(self, x):
        if x.ndim == 1:
            # If input is not batched, add a batch dimension
            x = x.unsqueeze(0)
        return self.amplitude(x)