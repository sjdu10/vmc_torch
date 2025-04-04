from autoray import numpy as np
import symmray as sr
from symmray.fermionic_local_operators import FermionicOperator
import torch
import numpy
import quimb.tensor as qtn
import autoray as ar
from autoray import do
from quimb.tensor.tensor_core import  *
from quimb.tensor.tensor_core import bonds, tags_to_oset, rand_uuid
from quimb.tensor.tensor_2d import Rotator2D, pairwise
from vmc_torch.global_var import DEBUG
import ast

#------Read PEPS from fTN model------
def get_psi_from_fTN(fTN_model):
    """
    Parameters
    ----------
    fTN_model : fTNModel
        fTN model to extract the wavefunction from.
    """
    # Reconstruct the original parameter structure (by unpacking from the flattened dict)
    params = {
        int(tid): {
            ast.literal_eval(sector): data
            for sector, data in blk_array.items()
        }
        for tid, blk_array in fTN_model.torch_tn_params.items()
    }
    # Reconstruct the TN with the new parameters
    psi = qtn.unpack(params, fTN_model.skeleton)
    return psi


#------Symmray function utils------
try:
    parse_edges_to_site_info = sr.utils.parse_edges_to_site_info
except:
    parse_edges_to_site_info = sr.parse_edges_to_site_info

def u1arr_to_z2arr(u1array):
    """
    Convert a FermionicArray with U1 symmetry to a FermionicArray with Z2 symmetry
    """
    def u1ind_to_z2indmap(u1indices):
        index_maps = []
        for blockind in u1indices:
            index_map = {}
            indicator = 0 #max value=blocind.size_total-1
            for c, dim in blockind.chargemap.items():
                for i in range(indicator, indicator+dim):
                    index_map[i]=int(c%2)
                indicator+=dim
            index_maps.append(index_map)
        return index_maps
    
    u1indices = u1array.indices
    u1charge = u1array.charge
    u1oddpos = u1array.oddpos
    u1duals = u1array.duals
    index_maps = u1ind_to_z2indmap(u1indices)
    z2array=sr.Z2FermionicArray.from_dense(u1array.to_dense(), index_maps=index_maps, duals=u1duals, charge=u1charge%2, oddpos=u1oddpos)
    return z2array

def u1peps_to_z2peps(peps):
    """
    Convert a PEPS with U1 symmetry to a PEPS with Z2 symmetry
    """
    pepsu1 = peps.copy()
    for ts in pepsu1.tensors:
        ts.modify(data=u1arr_to_z2arr(ts.data))
    return pepsu1.copy()

#------Amplitude Calculation------

class fPEPS(qtn.PEPS):
    def __init__(self, arrays, *, shape="urdlp", tags=None, site_ind_id="k{},{}", site_tag_id="I{},{}", x_tag_id="X{}", y_tag_id="Y{}", **tn_opts):
        super().__init__(arrays, shape=shape, tags=tags, site_ind_id=site_ind_id, site_tag_id=site_tag_id, x_tag_id=x_tag_id, y_tag_id=y_tag_id, **tn_opts)
        self.symmetry = self.arrays[0].symmetry
        self.spinless = True if self.phys_dim() == 2 else False
    
    def product_bra_state(self, config, reverse=1):
        product_tn = qtn.TensorNetwork()
        backend = self.tensors[0].data.backend
        dtype = eval(backend+'.'+self.tensors[0].data.dtype)
        if type(config) == numpy.ndarray:
            kwargs = {'like':config, 'dtype':dtype}
        elif type(config) == torch.Tensor:
            device = list(self.tensors[0].data.blocks.values())[0].device
            kwargs = {'like':config, 'device':device, 'dtype':dtype}
        if self.spinless:
            index_map = {0: 0, 1: 1}
            array_map = {
                0: do('array',[1.0,],**kwargs), 
                1: do('array',[1.0,],**kwargs)
            }
        else:
            if self.symmetry == 'Z2':
                index_map = {0:0, 1:1, 2:1, 3:0}
                array_map = {
                    0: do('array',[1.0, 0.0],**kwargs), 
                    1: do('array',[1.0, 0.0],**kwargs), 
                    2: do('array',[0.0, 1.0],**kwargs), 
                    3: do('array',[0.0, 1.0],**kwargs)
                }
            elif self.symmetry == 'U1':
                index_map = {0:0, 1:1, 2:1, 3:2}
                array_map = {
                    0: do('array',[1.0,],**kwargs), 
                    1: do('array',[1.0, 0.0],**kwargs), 
                    2: do('array',[0.0, 1.0],**kwargs), 
                    3: do('array',[1.0,],**kwargs)
                }
            elif self.symmetry == 'U1U1':
                index_map = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}
                array_map = {
                    0: do('array',[1.0],**kwargs),
                    1: do('array',[1.0],**kwargs), 
                    2: do('array',[1.0],**kwargs),
                    3: do('array',[1.0],**kwargs)
                }

        for n, site in zip(config, self.sites):
            p_ind = self.site_ind_id.format(*site)
            p_tag = self.site_tag_id.format(*site)
            tid = self.sites.index(site)

            n_charge = index_map[int(n)]
            n_array = array_map[int(n)]

            oddpos = None
            if not self.spinless:
                # assert self.symmetry == 'U1', "Only U1 symmetry is supported for spinful fermions for now."
                if int(n) == 1:
                    oddpos = (3*tid+1)*(-1)**reverse
                elif int(n) == 2:
                    oddpos = (3*tid+2)*(-1)**reverse
                elif int(n) == 3:
                    # oddpos = ((3*tid+1)*(-1)**reverse, (3*tid+2)*(-1)**reverse)
                    oddpos = None
            else:
                oddpos = (3*tid+1)*(-1)**reverse

            tsr_data = sr.FermionicArray.from_blocks(
                blocks={(n_charge,):n_array}, 
                duals=(True,),
                symmetry=self.symmetry, 
                charge=n_charge, 
                oddpos=oddpos
            )
            tsr = qtn.Tensor(data=tsr_data, inds=(p_ind,),tags=(p_tag, 'bra'))
            product_tn |= tsr

        return product_tn
    
    # NOTE: don't use @classmethod here, as we need to access the specific instance attributes
    def get_amp(self, config, inplace=False, conj=True, reverse=1, contract=True, efficient=True):
        """Get the amplitude of a configuration in a PEPS."""
        if efficient:
            return self.get_amp_efficient(config, inplace=inplace)
        peps = self if inplace else self.copy()
        product_state = self.product_bra_state(config, reverse=reverse).conj() if conj else self.product_bra_state(config, reverse=reverse)
        
        amp = peps|product_state # ---T---<---|n>

        if not contract:
            return amp
        
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
    
    def get_amp_efficient(self, config, inplace=False):
        """Slicing to get the amplitude, faster than contraction with a tensor product state."""
        peps = self if inplace else self.copy()
        backend = self.tensors[0].data.backend
        dtype = eval(backend + '.' + self.tensors[0].data.dtype)
        if type(config) == numpy.ndarray:
            kwargs = {'like': config, 'dtype': dtype}
        elif type(config) == torch.Tensor:
            device = list(self.tensors[0].data.blocks.values())[0].device
            kwargs = {'like': config, 'device': device, 'dtype': dtype}
        
        
        if self.spinless:
            raise NotImplementedError("Efficient amplitude calculation is not implemented for spinless fermions.")
        else:
            if self.symmetry == 'Z2':
                index_map = {0: 0, 1: 1, 2: 1, 3: 0}
                array_map = {
                    0: do('array', [1.0, 0.0], **kwargs),
                    1: do('array', [1.0, 0.0], **kwargs),
                    2: do('array', [0.0, 1.0], **kwargs),
                    3: do('array', [0.0, 1.0], **kwargs)
                }
            elif self.symmetry == 'U1':
                index_map = {0: 0, 1: 1, 2: 1, 3: 2}
                array_map = {
                    0: do('array', [1.0], **kwargs),
                    1: do('array', [1.0, 0.0], **kwargs),
                    2: do('array', [0.0, 1.0], **kwargs),
                    3: do('array', [1.0], **kwargs)
                }
            elif self.symmetry == 'U1U1':
                index_map = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}
                array_map = {
                    0: do('array',[1.0],**kwargs),
                    1: do('array',[1.0],**kwargs), 
                    2: do('array',[1.0],**kwargs),
                    3: do('array',[1.0],**kwargs)
                }
            

            for n, site in zip(config, self.sites):
                p_ind = peps.site_ind_id.format(*site)
                site_id = peps.sites.index(site)
                fts = peps.tensors[site_id]
                ftsdata = fts.data
                ftsdata.phase_sync(inplace=True) # explicitly apply all lazy phases that are stored and not yet applied
                phys_ind_order = fts.inds.index(p_ind)
                charge = index_map[int(n)]
                input_vec = array_map[int(n)]
                charge_sec_data_dict = ftsdata.blocks

                new_fts_inds = fts.inds[:phys_ind_order] + fts.inds[phys_ind_order + 1:]
                new_charge_sec_data_dict = {}
                for charge_blk, data in charge_sec_data_dict.items():
                    if charge_blk[phys_ind_order] == charge:
                        # new_data = data @ input_vec #BUG: This is not correct, should contract with the correct tensor index
                        new_data = do('tensordot', data, input_vec, axes=([phys_ind_order], [0]))
                        new_charge_blk = charge_blk[:phys_ind_order] + charge_blk[phys_ind_order + 1:]
                        new_charge_sec_data_dict[new_charge_blk] = new_data

                new_duals = ftsdata.duals[:phys_ind_order] + ftsdata.duals[phys_ind_order + 1:]

                if int(n) == 1:
                    new_oddpos = (3 * site_id + 1) * (-1)
                elif int(n) == 2:
                    new_oddpos = (3 * site_id + 2) * (-1)
                elif int(n) == 3 or int(n) == 0:
                    new_oddpos = ()

                new_oddpos1 = FermionicOperator(new_oddpos, dual=True) if new_oddpos != () else ()
                new_oddpos = ftsdata.oddpos + (new_oddpos1,) if new_oddpos1 is not () else ftsdata.oddpos
                oddpos = list(new_oddpos)[::-1]
                try:
                    if ftsdata.symmetry == 'U1':
                        new_charge = charge + ftsdata.charge
                    elif ftsdata.symmetry == 'Z2':
                        new_charge = (charge + ftsdata.charge) % 2 # Z2 symmetry, charge should be 0 or 1
                    elif ftsdata.symmetry == 'U1U1':
                        new_charge = (charge[0] + ftsdata.charge[0], charge[1] + ftsdata.charge[1]) # U1U1 symmetry, charge should be a tuple of two integers
                    new_fts_data = sr.FermionicArray.from_blocks(new_charge_sec_data_dict, duals=new_duals, charge=new_charge, oddpos=oddpos, symmetry=ftsdata.symmetry)
                except:
                    print(n, site, phys_ind_order, charge_sec_data_dict, new_charge_sec_data_dict)
                fts.modify(data=new_fts_data, inds=new_fts_inds, left_inds=None)

            amp = qtn.PEPS(peps)

            return amp


def generate_random_fpeps(Lx, Ly, D, seed, symmetry='Z2', Nf=0, cyclic=False, spinless=False):
    """Generate a random spinless/spinful fermionic square PEPS of shape (Lx, Ly)."""

    assert symmetry in ['Z2', 'U1', 'U1U1'], "Only Z2 ,U1 and U1U1 symmetries are supported."
    
    edges = qtn.edges_2d_square(Lx, Ly, cyclic=cyclic)
    site_info = parse_edges_to_site_info(
        edges,
        D,
        phys_dim=2 if spinless else 4,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
    )

    peps = qtn.TensorNetwork()
    rng = np.random.default_rng(seed)
    charge_config = np.zeros(Lx*Ly, dtype=int)

    # generate a random binary string with Nf ones in it
    if symmetry == 'U1':
        if spinless:
            charge_config[:Nf] = 1
            rng.shuffle(charge_config)
        else:
            charge_config_netket = from_quimb_config_to_netket_config(charge_config)
            charge_config_netket[:Nf] = 1
            rng.shuffle(charge_config_netket)
            charge_config = from_spinful_ind_to_charge(from_netket_config_to_quimb_config(charge_config_netket))

    elif symmetry == 'Z2':
        parity_config = charge_config
    
    elif symmetry == 'U1U1': # Sz=0
        nu, nd = int(Nf/2), int(Nf/2)
        charge_config_netket = from_quimb_config_to_netket_config(charge_config)
        charge_config_netket_u = charge_config_netket[:len(charge_config_netket)//2]  # up spins
        charge_config_netket_d = charge_config_netket[len(charge_config_netket)//2:]  # down spins
        # put nu 1s in the first half of the configuration (up spins) and shuffle
        charge_config_netket_u[:nu] = 1  # assign nu up spins
        rng.shuffle(charge_config_netket_u)  # shuffle the up spins to randomize their positions
        # put nd 1s in the second half of the configuration (down spins) and shuffle
        charge_config_netket_d[:nd] = 1  # assign nd down spins
        rng.shuffle(charge_config_netket_d)  # shuffle the down spins to randomize their positions
        # combine the up and down configurations back into a single netket configuration
        charge_config_netket = np.concatenate((charge_config_netket_u, charge_config_netket_d))
        charge_config = from_spinful_ind_to_charge(from_netket_config_to_quimb_config(charge_config_netket), symmetry='U1U1')

    for site, info in sorted(site_info.items()):
        tid = site[0] * Ly + site[1]

        # virtual index charge distribution
        if symmetry == 'Z2':
            block_indices = [
                sr.BlockIndex({0: d // 2, 1: d // 2}, dual=dual)
                for d, dual in zip(info["shape"][:-1], info["duals"][:-1])
            ]
        elif symmetry == 'U1':
            block_indices = [
                sr.BlockIndex({0: d // 4, 1: d // 2, 2: d // 4}, dual=dual)
                for d, dual in zip(info["shape"][:-1], info["duals"][:-1])
            ]
        elif symmetry == 'U1U1':
            block_indices = [
                sr.BlockIndex({(0, 0): d//4, (0, 1): d//4, (1, 0): d//4, (1, 1): d//4}, dual=dual)
                for d, dual in zip(info["shape"][:-1], info["duals"][:-1])
            ]

        # physical index charge distribution
        p = info['shape'][-1]

        if symmetry == 'Z2':
            block_indices.append(
                sr.BlockIndex({0: p // 2, 1: p // 2}, dual=info["duals"][-1])
            )
        elif symmetry == 'U1':
            if spinless:
                block_indices.append(
                sr.BlockIndex({0: p // 2, 1: p // 2}, dual=info["duals"][-1])
            )
            else:
                block_indices.append(
                    sr.BlockIndex({0: p // 4, 1: p // 2, 2: p // 4}, dual=info["duals"][-1])
                )
        elif symmetry == 'U1U1':
            block_indices.append(
                sr.BlockIndex({(0, 0): p//4, (0, 1): p//4, (1, 0): p//4, (1, 1): p//4}, dual=info["duals"][-1])
            )
        
        # random fermionic array
        if symmetry == 'Z2':
            data = sr.Z2FermionicArray.random(
                block_indices,
                charge=1 if parity_config[tid] else 0,
                seed=rng,
                oddpos=3*tid,
            )
        elif symmetry == 'U1':
            data = sr.U1FermionicArray.random(
                block_indices,
                charge=int(charge_config[tid]),
                seed=rng,
                oddpos=3*tid,
            )
        elif symmetry == 'U1U1':
            data = sr.U1U1FermionicArray.random(
                block_indices,
                charge=charge_config[tid],
                seed=rng,
                oddpos=3*tid,
            )

        peps |= qtn.Tensor(
            data=data,
            inds=info["inds"],
            tags=info["tags"],
        )

    # required to view general TN as an actual PEPS
    for i, j in site_info:
        peps[f"I{i},{j}"].add_tag([f"X{i}", f"Y{j}"])

    peps.view_as_(
        fPEPS,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        Lx=Lx,
        Ly=Ly,
    )
    peps = peps.copy() # set symmetry during initialization
    assert peps.spinless == spinless

    return peps, charge_config


def product_bra_state(psi, config, check=False,reverse=True, dualness=True):
    #XXX: need to be deleted in the future
    raise DeprecationWarning(
        'This function will be deprecated in favor of `fPEPS.product_bra_state` method. Please use `fPEPS` instead.'
    )
    product_tn = qtn.TensorNetwork()
    backend = psi.tensors[0].data.backend
    device = config.device
    dtype = eval(backend+'.'+psi.tensors[0].data.dtype)
    if type(config) == numpy.ndarray:
        kwargs = {'like':config, 'dtype':dtype}
    elif type(config) == torch.Tensor:
        kwargs = {'like':config, 'device':device, 'dtype':dtype}
    if psi.spinless:
        index_map = {0: 0, 1: 1}
        array_map = {
            0: do('array',[1.0],**kwargs), 
            1: do('array',[1.0],**kwargs)
        }
    else:
        if psi.symmetry == 'Z2':
            index_map = {0:0, 1:1, 2:1, 3:0}
            array_map = {
                0: do('array',[1.0, 0.0],**kwargs), 
                1: do('array',[1.0, 0.0],**kwargs), 
                2: do('array',[0.0, 1.0],**kwargs), 
                3: do('array',[0.0, 1.0],**kwargs)
            }
        elif psi.symmetry == 'U1':
            index_map = {0:0, 1:1, 2:1, 3:2}
            array_map = {
                0: do('array',[1.0],**kwargs), 
                1: do('array',[1.0, 0.0],**kwargs), 
                2: do('array',[0.0, 1.0],**kwargs), 
                3: do('array',[1.0],**kwargs)
            }

    iter = zip(config, psi.sites) # if not reverse else zip(config[::-1], psi.sites[::-1])

    for n, site in iter:
        p_ind = psi.site_ind_id.format(*site)
        p_tag = psi.site_tag_id.format(*site)
        tid = psi.sites.index(site)

        n_charge = index_map[int(n)]
        n_array = array_map[int(n)]

        oddpos = None
        if not psi.spinless:
            if int(n) == 1:
                oddpos = (3*tid+1)*(-1)**reverse
            elif int(n) == 2:
                oddpos = (3*tid+2)*(-1)**reverse
            elif int(n) == 3:
                # oddpos = ((3*tid+1)*(-1)**reverse, (3*tid+2)*(-1)**reverse)
                oddpos = None
        else:
            oddpos = (3*tid+1)*(-1)**reverse
        
        tsr_data = sr.FermionicArray.from_blocks(
            blocks={(n_charge,):n_array}, 
            duals=(dualness,),
            symmetry=psi.symmetry, 
            charge=n_charge, 
            oddpos=oddpos
        )
        
        if check:
            if int(n)==0:
                blocks = {(0,): do('array',[1.0],like=config,dtype=dtype,device=device), (1,): do('array',[0.0, 0.0],like=config,dtype=dtype,device=device), (2,):do('array',[0.0],like=config,dtype=dtype,device=device)}
            elif int(n)==1:
                blocks = {(0,): do('array',[0.0],like=config,dtype=dtype,device=device), (1,): do('array',[1.0, 0.0],like=config,dtype=dtype,device=device), (2,):do('array',[0.0],like=config,dtype=dtype,device=device)}
            elif int(n)==2:
                blocks = {(0,): do('array',[0.0],like=config,dtype=dtype,device=device), (1,): do('array',[0.0, 1.0],like=config,dtype=dtype,device=device), (2,):do('array',[0.0],like=config,dtype=dtype,device=device)}
            elif int(n)==3:
                blocks = {(0,): do('array',[0.0],like=config,dtype=dtype,device=device), (1,): do('array',[0.0, 0.0],like=config,dtype=dtype,device=device), (2,):do('array',[1.0],like=config,dtype=dtype,device=device)}
            tsr_data = sr.FermionicArray.from_blocks(
                blocks=blocks, 
                duals=(dualness,),
                symmetry=psi.symmetry, 
                charge=n_charge, 
                oddpos=oddpos
            )
        
        tsr = qtn.Tensor(data=tsr_data, inds=(p_ind,),tags=(p_tag, 'bra', f'X{site[0]}', f'Y{site[1]}'))
        product_tn |= tsr

    product_tn.view_as_(
        qtn.PEPS,
        site_ind_id="k{},{}",
        site_tag_id="I{},{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        Lx=psi.Lx,
        Ly=psi.Ly,
    )

    return product_tn

def get_amp(peps, config, inplace=False, conj=True):
    """Get the amplitude of a configuration in a PEPS."""
    #XXX: need to be deleted in the future
    raise DeprecationWarning(
        "The function `get_amp` in `fermion_utils` is deprecated, please use `fPEPS.get_amp` instead."
    )
    if not inplace:
        peps = peps.copy()
    bra = product_bra_state(peps, config, dualness=False).conj() if conj else product_bra_state(peps, config, dualness=False)

    amp = bra|peps
    
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

class fMPS(qtn.MatrixProductState):
    def __init__(
        self,
        arrays,
        *,
        sites=None,
        L=None,
        shape="lrp",
        tags=None,
        site_ind_id="k{}",
        site_tag_id="I{}",
        **tn_opts,
    ):
        super().__init__(arrays, sites=sites, L=L, shape=shape, tags=tags, site_ind_id=site_ind_id, site_tag_id=site_tag_id, **tn_opts)

        self.symmetry = self.arrays[0].symmetry
        try:
            self.spinless = True if self.phys_dim() == 2 else False
        except KeyError:
            self.spinless = True if self.ind_size(self.site_ind_id.format(self.L-1)) == 2 else False
    
    def product_bra_state(self, config, reverse=1):
        """For product state ALWAYS make sure the set of oddposes are different from the set of oddposes in the TNS |psi>.
        When using some overlapping oddposes, the computation of the amplitude will gain some unphysical global phase!!"""
        product_tn = qtn.TensorNetwork()
        backend = self.tensors[0].data.backend
        dtype = eval(backend+'.'+self.tensors[0].data.dtype)
        if type(config) == numpy.ndarray:
            kwargs = {'like':config, 'dtype':dtype}
        elif type(config) == torch.Tensor:
            device = list(self.tensors[0].data.blocks.values())[0].device
            kwargs = {'like':config, 'device':device, 'dtype':dtype}
        if self.spinless:
            index_map = {0: 0, 1: 1}
            array_map = {
                0: do('array',[1.0,],**kwargs), 
                1: do('array',[1.0,],**kwargs)
            }
        else:
            if self.symmetry == 'Z2':
                index_map = {0:0, 1:1, 2:1, 3:0}
                array_map = {
                    0: do('array',[1.0, 0.0],**kwargs), 
                    1: do('array',[1.0, 0.0],**kwargs), 
                    2: do('array',[0.0, 1.0],**kwargs), 
                    3: do('array',[0.0, 1.0],**kwargs)
                }
            elif self.symmetry == 'U1':
                index_map = {0:0, 1:1, 2:1, 3:2}
                array_map = {
                    0: do('array',[1.0,],**kwargs), 
                    1: do('array',[1.0, 0.0],**kwargs), 
                    2: do('array',[0.0, 1.0],**kwargs), 
                    3: do('array',[1.0,],**kwargs)
                }

        for n, site in zip(config, self.sites):
            p_ind = self.site_ind_id.format(site)
            p_tag = self.site_tag_id.format(site)
            tid = self.sites.index(site)

            n_charge = index_map[int(n)]
            n_array = array_map[int(n)]

            oddpos = None
            if not self.spinless:
                # assert self.symmetry == 'U1', "Only U1 symmetry is supported for spinful fermions for now."
                if int(n) == 1:
                    oddpos = (3*tid+1)*(-1)**reverse
                elif int(n) == 2:
                    oddpos = (3*tid+2)*(-1)**reverse
                elif int(n) == 3:
                    # oddpos = ((3*tid+1)*(-1)**reverse, (3*tid+2)*(-1)**reverse)
                    oddpos = None
            else:
                oddpos = (3*tid+1)*(-1)

            tsr_data = sr.FermionicArray.from_blocks(
                blocks={(n_charge,):n_array}, 
                duals=(True,),
                symmetry=self.symmetry, 
                charge=n_charge, 
                oddpos=oddpos
            )
            tsr = qtn.Tensor(data=tsr_data, inds=(p_ind,),tags=(p_tag))
            product_tn |= tsr

        return product_tn
    
    # NOTE: don't use @classmethod here, as we need to access the specific instance attributes
    def get_amp(self, config, inplace=False, conj=True, efficient=True):
        """Get the amplitude of a configuration in a PEPS."""
        if efficient:
            return self.get_amp_efficient(config, inplace=inplace)
        mps = self if inplace else self.copy()
        product_state = self.product_bra_state(config).conj() if conj else self.product_bra_state(config)
        
        amp = mps|product_state # ---T---<---|n>
        
        for site in mps.sites:
            site_tag = mps.site_tag_id.format(site)
            amp.contract_(tags=site_tag)

        amp.view_as_(
            qtn.MatrixProductState,
            site_ind_id="k{}",
            site_tag_id="I{}",
            L=mps.L,
            cyclic=mps.cyclic,
        )
        # if DEBUG:
        #     index_map = {0:0, 1:1, 2:1, 3:2}
        #     amp_efficient = self.get_amp_efficient(config)
        #     # print('State charge:', [ts.data.charge for ts in mps.tensors])
        #     # print('Config charge:', [index_map[int(n)] for n in config])
        #     # print('Original oddpos:', [ts.data.oddpos for ts in mps.tensors])
        #     # print('Correct oddpos:', [ts.data.oddpos for ts in amp.tensors])
        #     # print('Incorrect oddpos:', [ts.data.oddpos for ts in amp_efficient.tensors])
        #     print(config, amp.contract(), amp_efficient.contract())
        #     # for i in range(len(amp.tensors)):
        #     #     # print(amp.tensors[i].data.blocks,'   ', amp_efficient.tensors[i].data.blocks, '\n')
        #     #     print(amp.tensors[i].data,'   ', amp_efficient.tensors[i].data, '\n')
        #     print('------------------------------------')
        #     print(amp.tensor_map)
        #     print('------------------------------------')
        #     print(amp_efficient.tensor_map)
        #     print('------------------------------------')
        #     def compare_objects(obj1, obj2):
        #         # Check if the objects are of the same type
        #         if type(obj1) != type(obj2):
        #             return False
                
        #         for sec in obj1.blocks.keys():
        #             if (obj1.blocks[sec] != obj2.blocks[sec]).any():
        #                 print('Mismatch in blocks:', sec)
        #                 print(obj1.blocks[sec], obj2.blocks[sec])
        #                 return False
        #         if obj1.duals != obj2.duals:
        #             print('Mismatch in duals')
        #             print(obj1.duals, obj2.duals)
        #             return False
        #         if obj1.charge != obj2.charge:
        #             print('Mismatch in charge')
        #             print(obj1.charge, obj2.charge)
        #             return False
        #         if obj1.oddpos != obj2.oddpos:
        #             print('Mismatch in oddpos')
        #             print(obj1.oddpos, obj2.oddpos)
        #             return False
        #         if obj1.symmetry != obj2.symmetry:
        #             print(obj1.symmetry, obj2.symmetry)
        #             return False
        #         return True

        #         # return True
        #     for i in range(len(amp.tensors)):
        #         print(compare_objects(amp.tensors[i].data, amp_efficient.tensors[i].data))
        #         # print(amp.tensors[i].data.__dict__ == amp_efficient.tensors[i].data.__dict__)
        #         # print(amp.tensors[i].data,'   ', amp_efficient.tensors[i].data, '\n')
        return amp
    
    def get_amp_efficient(self, config, inplace=False):
        """Slicing to get the amplitude, faster than contraction with a tensor product state."""
        mps = self if inplace else self.copy()
        backend = self.tensors[0].data.backend
        dtype = eval(backend+'.'+self.tensors[0].data.dtype)
        if type(config) == numpy.ndarray:
            kwargs = {'like':config, 'dtype':dtype}
        elif type(config) == torch.Tensor:
            device = list(self.tensors[0].data.blocks.values())[0].device
            kwargs = {'like':config, 'device':device, 'dtype':dtype}
        if self.spinless:
            raise NotImplementedError("Efficient amplitude calculation is not implemented for spinless fermions.")
            # index_map = {0: 0, 1: 1}
            # array_map = {
            #     0: do('array',[1.0,],**kwargs), 
            #     1: do('array',[1.0,],**kwargs)
            # }
        else:
            if self.symmetry == 'Z2':
                index_map = {0:0, 1:1, 2:1, 3:0}
                array_map = {
                    0: do('array',[1.0, 0.0],**kwargs), 
                    1: do('array',[1.0, 0.0],**kwargs), 
                    2: do('array',[0.0, 1.0],**kwargs), 
                    3: do('array',[0.0, 1.0],**kwargs)
                }
            elif self.symmetry == 'U1':
                index_map = {0:0, 1:1, 2:1, 3:2}
                array_map = {
                    0: do('array',[1.0,],**kwargs), 
                    1: do('array',[1.0, 0.0],**kwargs), 
                    2: do('array',[0.0, 1.0],**kwargs), 
                    3: do('array',[1.0,],**kwargs)
                }
        
            for n, site in zip(config, self.sites):
                p_ind = mps.site_ind_id.format(site)
                tid = mps.sites.index(site)
                fts = mps[tid]
                ftsdata = fts.data
                ftsdata.phase_sync(inplace=True) # explicitly apply all lazy phases that are stored and not yet applied
                phys_ind_order = fts.inds.index(p_ind)
                charge = index_map[int(n)]
                input_vec = array_map[int(n)]
                charge_sec_data_dict = ftsdata.blocks
                new_fts_inds = fts.inds[:phys_ind_order] + fts.inds[phys_ind_order+1:]
                new_charge_sec_data_dict = {}
                for charge_blk, data in charge_sec_data_dict.items():
                    if charge_blk[phys_ind_order] == charge:
                        new_data = do('tensordot', data, input_vec, axes=([phys_ind_order], [0]))
                        new_charge_blk = charge_blk[:phys_ind_order] + charge_blk[phys_ind_order+1:]
                        new_charge_sec_data_dict[new_charge_blk]=new_data
                        
                new_duals = ftsdata.duals[:phys_ind_order] + ftsdata.duals[phys_ind_order+1:]

                if int(n) == 1:
                    new_oddpos = (3*tid+1)*(-1)
                elif int(n) == 2:
                    new_oddpos = (3*tid+2)*(-1)
                elif int(n) == 3 or int(n) == 0:
                    new_oddpos = ()

                new_oddpos1 = FermionicOperator(new_oddpos, dual=True) if new_oddpos is not () else ()
                new_oddpos = ftsdata.oddpos + (new_oddpos1,) if new_oddpos1 is not () else ftsdata.oddpos
                oddpos = list(new_oddpos)[::-1]
                
                new_fts_data = sr.FermionicArray.from_blocks(new_charge_sec_data_dict, duals=new_duals, charge=charge+ftsdata.charge, oddpos=oddpos, symmetry=ftsdata.symmetry)
                fts.modify(data=new_fts_data, inds=new_fts_inds, left_inds=None)

            amp = qtn.MatrixProductState(mps)

            return amp

class fMPS_TNF(fMPS):
    def __init__(self, arrays, depth=None, L=None, *args, **kwargs):
        # short-circuit for copying TNFs
        if isinstance(arrays, fMPS_TNF):
            self.Lx = arrays.Lx
            self.Ly = arrays.Ly
            super().__init__(arrays, *args, **kwargs)
            return
        self.Lx = depth+1
        self.Ly = L
        super().__init__(arrays, *args, **kwargs)
    
    def set_Lx(self, Lx):
        self.Lx = Lx
    def set_Ly(self, Ly):
        self.Ly = Ly
    # NOTE: don't use @classmethod here, as we need to access the specific instance attributes
    def get_amp(self, config, inplace=False, conj=True, reverse=1):
        """Get the amplitude of a configuration in a PEPS."""
        tnf = self if inplace else self.copy()
        product_state = self.product_bra_state(config, reverse=reverse).conj() if conj else self.product_bra_state(config, reverse=reverse)
        
        amp = tnf|product_state # ---T---<---|n>
        
        for ind in tnf.site_inds:
            amp.contract_ind(ind)

        amp.view_as_(
            qtn.PEPS,
            site_ind_id="k{}",
            x_tag_id="ROUND_{}",
            y_tag_id="I{}", 
            site_tag_id="I{},{}",
            Lx=tnf.Lx,
            Ly=tnf.Ly,
        )
        return amp

def generate_random_fmps(L, D, seed, symmetry='Z2', Nf=0, cyclic=False, spinless=False):
    """Generate a random spinless/spinful fermionic MPS of length L."""
    assert symmetry == 'Z2' or symmetry == 'U1', "Only Z2 and U1 symmetries are supported."

    edges = qtn.edges_1d_chain(L, cyclic=cyclic)
    site_info = parse_edges_to_site_info(
        edges,
        D,
        phys_dim=2 if spinless else 4,
        site_ind_id="k{}",
        site_tag_id="I{}",
    )

    mps = qtn.TensorNetwork()
    rng = np.random.default_rng(seed)
    charge_config = np.zeros(L, dtype=int)

    # generate a random binary string with Nf ones in it
    if symmetry == 'U1':
        if spinless:
            charge_config[:Nf] = 1
            rng.shuffle(charge_config)
        else:
            charge_config_netket = from_quimb_config_to_netket_config(charge_config)
            charge_config_netket[:Nf] = 1
            rng.shuffle(charge_config_netket)
            charge_config = from_spinful_ind_to_charge(from_netket_config_to_quimb_config(charge_config_netket))

    elif symmetry == 'Z2':
        parity_config = charge_config

    for site, info in sorted(site_info.items()):
        tid = site
        # bond index charge distribution
        block_indices = [
            sr.BlockIndex({0: d // 2, 1: d // 2}, dual=dual)
            for d, dual in zip(info["shape"][:-1], info["duals"][:-1])
        ]
        # physical index
        p = info['shape'][-1]
        if symmetry == 'Z2':
            block_indices.append(
                sr.BlockIndex({0: p // 2, 1: p // 2}, dual=info["duals"][-1])
            )
        elif symmetry == 'U1':
            if spinless:
                block_indices.append(
                sr.BlockIndex({0: p // 2, 1: p // 2}, dual=info["duals"][-1])
            )
            else:
                block_indices.append(
                    sr.BlockIndex({0: p // 4, 1: p // 2, 2: p // 4}, dual=info["duals"][-1])
                )
        
        # random fermionic array
        if symmetry == 'Z2':
            data = sr.Z2FermionicArray.random(
                block_indices,
                charge=1 if parity_config[tid] else 0,
                seed=rng,
                oddpos=3*tid,
            )
        elif symmetry == 'U1':
            data = sr.U1FermionicArray.random(
                block_indices,
                charge=int(charge_config[tid]),
                seed=rng,
                oddpos=3*tid,
            )
        
        mps |= qtn.Tensor(
            data=data,
            inds=info["inds"],
            tags=info["tags"],
        )

    # required to view general TN as an actual PEPS

    mps.view_as_(
        fMPS,
        L=L,
        site_ind_id="k{}",
        site_tag_id="I{}",
        cyclic=cyclic,
    )
    mps = mps.copy() # set symmetry during initialization
    return mps, charge_config


def form_gated_fmps_tnf(
        fmps, 
        ham, 
        depth,
        tau = 0.5,
        nn_where_list=None,
        x_tag_id="ROUND_{}",
        y_tag_id="I{}",
        site_tag_id="I{},{}",
    ):
    fmps1 = fmps.copy()

    if not isinstance(nn_where_list, list):
        Warning("nn_where_list is not a list of tuples, using all nearest neighbor terms in the Hamiltonian")
        nn_where_list = [(i, i+1) for i in range(fmps.L-1)]
    
    # Change tags for the initial MPS
    for ts in fmps1.tensors:
        ts.modify(tags=['ROUND_0']+list(ts.tags))
    
    # Apply the gates and add corresponding tags
    for i in range(depth):
        for where in nn_where_list:
            gate = ham.get_gate_expm(where, -1*tau)
            site_inds = [fmps1.site_ind_id.format(site) for site in where]
            extra_tags = ['ROUND_{}'.format(i+1)]
            ltag = fmps1.site_tag_id.format(where[0])
            rtag = fmps1.site_tag_id.format(where[1])
            fmps1 = fmps1.gate_inds(gate, inds=site_inds, contract='split-gate', tags=extra_tags, ltags=ltag, rtags=rtag)
    
    # Contract the gates in each round to a MPO
    for i in range(1,depth+1):
        for site in fmps1.sites:
            fmps1.contract_tags_([fmps1.site_tag_id.format(site), f'ROUND_{i}'], inplace=True, which='all')
    
    # Add site tags
    for x in range(0,depth+1):
        for y in range(fmps1.L):
            ts = fmps1[[x_tag_id.format(x), y_tag_id.format(y)]]
            ts.add_tag(site_tag_id.format(x,y))
    
    fmps1 = fMPS_TNF.from_TN(fmps1)
    fmps1.set_Lx(depth+1)
    fmps1.set_Ly(fmps1.L)

    return fmps1


# --- Utils for calculating global phase on product states ---

def get_spinful_parity_map():
    return {0:0, 1:1, 2:1, 3:0}

def get_spinful_charge_map(symmetry='U1'):
    if symmetry == 'Z2':
        return get_spinful_parity_map()
    elif symmetry == 'U1':
        return {0:0, 1:1, 2:1, 3:2}
    elif symmetry == 'U1U1':
        return {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}
    else:
        raise ValueError(f"Symmetry {symmetry} is not supported for spinful charge mapping.")

def from_spinful_ind_to_charge(config, symmetry='U1'):
    charge_map = get_spinful_charge_map(symmetry)
    return np.array([charge_map[n] for n in config])

def from_netket_config_to_quimb_config(netket_configs):
    """Translate netket spin-1/2 fermion config to tensor network product state config"""
    def func(netket_config):
        total_sites = len(netket_config)//2
        spin_up = netket_config[:total_sites]
        spin_down = netket_config[total_sites:]
        sum_spin = spin_up + spin_down
        quimb_config = np.zeros(total_sites, dtype=int)
        for i in range(total_sites):
            if sum_spin[i] == 0:
                quimb_config[i] = 0
            if sum_spin[i] == 2:
                quimb_config[i] = 3
            if sum_spin[i] == 1:
                if spin_down[i] == 1:
                    quimb_config[i] = 1
                else:
                    quimb_config[i] = 2
        return quimb_config
    if len(netket_configs.shape) == 1:
        return func(netket_configs)
    else:
        # batched
        return np.array([func(netket_config) for netket_config in netket_configs])

def from_quimb_config_to_netket_config(quimb_config):
    """Translate tensor network product state config to netket spin-1/2 config"""

    def func(quimb_config):
        total_sites = len(quimb_config)
        spin_up = np.zeros(total_sites, dtype=int)
        spin_down = np.zeros(total_sites, dtype=int)
        for i in range(total_sites):
            if quimb_config[i] == 0:
                spin_up[i] = 0
                spin_down[i] = 0
            if quimb_config[i] == 1:
                spin_up[i] = 0
                spin_down[i] = 1
            if quimb_config[i] == 2:
                spin_up[i] = 1
                spin_down[i] = 0
            if quimb_config[i] == 3:
                spin_up[i] = 1
                spin_down[i] = 1
        return np.concatenate((spin_up, spin_down))
    if len(quimb_config.shape) == 1:
        return func(quimb_config)
    else:
        # batched
        return np.array([func(quimb_config) for quimb_config in quimb_config])

def detect_hopping(configi, configj):
    """Detect the hopping between two configurations"""
    site_ls = np.asarray(configi-configj).nonzero()[0]
    if len(site_ls) == 2:
        return site_ls
    else:
        return None

def calc_phase_netket(configi, configj):
    """Calculate the phase factor for the matrix element in the netket basis, where all spin-up spins are placed before all spin-down spins.
    Globally, this convention gives a spin configuration (|uuu...ddd...>) when the configuration is flattened to 1D.
    Physically, the basis is written as a^d_1u a^d_2u ... a^d_nu a^d_1d a^d_2d ... a^d_nd |0>."""
    hopping = detect_hopping(configi, configj)
    if hopping is not None:
        netket_config_i = from_quimb_config_to_netket_config(configi)
        netket_config_j = from_quimb_config_to_netket_config(configj)
        netket_site_i, netket_site_j = (netket_config_i - netket_config_j).nonzero()[0]
        phase = (-1)**(sum(netket_config_i[netket_site_i+1:netket_site_j])%2)
        return phase
    else:
        return 1

def calc_phase_symmray(configi, configj):
    """Calculate the operator matrix element phase in symmray, assuming local basis convention for spinful fermion is (|up, down>).
    Globally, this convention gives a staggered spin configuration (|dududu...>) when the configuration is flattened to 1D.
    Physically, the basis is written as a^d_1d a^d_1u... a^d_nu a^d_nd |0>."""
    hopping = detect_hopping(configi, configj)
    if hopping is not None:
        netket_config_i = from_quimb_config_to_netket_config(configi)
        netket_config_j = from_quimb_config_to_netket_config(configj)
        config_i_spin_up =netket_config_i[:len(netket_config_i)//2]
        config_i_spin_down =netket_config_i[len(netket_config_i)//2:]
        config_j_spin_up =netket_config_j[:len(netket_config_j)//2]
        config_j_spin_down =netket_config_j[len(netket_config_j)//2:]
        symmray_config_i = tuple()
        symmray_config_j = tuple()
        for i in range(len(config_i_spin_up)):
            symmray_config_i += (config_i_spin_down[i], config_i_spin_up[i])
            symmray_config_j += (config_j_spin_down[i], config_j_spin_up[i])
        symmray_site_i, symmray_site_j = (np.asarray(symmray_config_i) - np.asarray(symmray_config_j)).nonzero()[0]
        phase = (-1)**(sum(symmray_config_i[symmray_site_i+1:symmray_site_j])%2)
        return phase
    else:
        return 1

def calc_phase_correction_netket_symmray(configi, configj):
    phase_netket = calc_phase_netket(configi, configj)
    phase_symmray = calc_phase_symmray(configi, configj)
    return phase_netket/phase_symmray


#------projector insertion------

def _dual_reverse_array(array, inplace=False):
    """Reverse the dualness of all the indices of an AbelianArray. 
    Safe only when the net charge of the array is 0 and when you know what you're doing."""
    if not inplace:
        array = array.copy()
    new_indices = tuple(ix.conj() for ix in array._indices)
    array._indices = new_indices
    return array

def insert_compressor(tn, ltags, rtags, new_ltags=None, new_rtags=None, max_bond=4, inplace=False, draw_tn=False):
    """Insert a compressor between two sets of tags."""
    amp_x = tn.copy() if not inplace else tn
    tn_backup = tn.copy()

    amp2, lrtn = amp_x.partition(ltags+rtags, inplace=False)
    rtn, ltn = lrtn.partition(ltags, inplace=False)
    lts = ltn.contract()
    rts = rtn.contract()

    bix = bonds(lts, rts)
    left_inds = tags_to_oset(lts.inds) - bix
    right_inds = tags_to_oset(rts.inds) - bix


    Ql,Rl = lts.split(left_inds=left_inds, get='tensors', method='qr')
    Rr,Qr = rts.split(left_inds=bix, get='tensors', method='lq')

    M = (Rl&Rr).contract()
    M.drop_tags()

    U, s, Vd = M.split(
        left_inds=Ql.bonds(M),
        get='tensors',
        absorb=None,
        method='svd',
        max_bond=max_bond,
        cutoff=0.0,
    )

    U_array = U.data
    Vd_array = Vd.data
    # absorb the singular values block by block (only works for blocked matrices)
    for c0, c1 in U_array.sectors:
        s_sqrt_inv = 1 / ar.do('sqrt', s.data.blocks[c1])
        # Torch back propagation does not allow in-place operation like x*=y, must use out-of-place operation x=x*y for future gradient computation.
        U_array.blocks[(c0,c1)] = U_array.blocks[(c0,c1)] * s_sqrt_inv.reshape((1, -1)) 
        Vd_array.blocks[(c0,c1)] = Vd_array.blocks[(c0,c1)] * s_sqrt_inv.reshape((-1, 1))

    U.modify(data=U_array.conj())
    Vd.modify(data=Vd_array.conj())


    Pl = (Rr|Vd).contract()
    Pr = (U|Rl).contract()
    
    new_bix_left = tags_to_oset([rand_uuid(),rand_uuid()])
    new_bix_right = tags_to_oset([rand_uuid(),rand_uuid()])
    Pl.reindex_(dict(zip(bix, new_bix_left)))
    Pr.reindex_(dict(zip(bix, new_bix_right)))
    Pl.drop_tags()
    Pr.drop_tags()
    new_ltags = tags_to_oset(new_ltags) | tags_to_oset('proj')# | tags_to_oset(ltags)
    new_rtags = tags_to_oset(new_rtags) | tags_to_oset('proj')# | tags_to_oset(rtags)
    Pl.add_tag(new_ltags)
    Pr.add_tag(new_rtags)

    ltn.reindex_(dict(zip(bix, new_bix_left)))
    rtn.reindex_(dict(zip(bix, new_bix_right)))
    tn0 = (amp2|ltn|Pr|Pl|rtn)
    
    if draw_tn:
        tn0.draw(color=['proj'], figsize=(12, 12))

    return tn0

def insert_proj_peps(amp, max_bond, yrange, xrange=None, from_which='ymin', lazy=False):
    """Insert projectors in a PEPS along the x direction towards y direction."""
    if yrange is None:
        yrange = [0, amp.Ly-1]
    r = Rotator2D(amp, xrange=xrange, yrange=yrange, from_which=from_which)
    tn_calc = amp.copy()
    for i, inext in pairwise(r.sweep):
        i_passed = [x for x in range(i)]
        # we compute the projectors from an untouched copy
        tn_calc = tn_calc.copy()
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
                    max_bond=max_bond,
                )
        # tn_calc.draw(color='proj')
        
        if not lazy:
            # contract each pair of boundary tensors with their projectors
            for j in r.sweep_other:
                tn_calc.contract_tags_(
                    (r.site_tag(i, j), r.site_tag(inext, j)),
                )

    return tn_calc


def flatten_proj_params(params):
    """Flatten the tensor parameters into a vector.
    
    Parameters
    ----------
    params : dict
        Dictionary containing the projector tensor parameters.
    
    Returns
    ----------
    vec_proj : array
        Flattened projector tensor parameters.
    """
    vec_proj = []
    for tid, ts_values in params.items():
        for blk, data in ts_values.items():
            vec_proj += list(data.flatten())

    return do('stack', vec_proj)


def reconstruct_proj_params(vec_params, params):
    """Reconstruct the tensor parameters from a flattened vector.
    
    Parameters
    ----------
    vec_proj : array
        Flattened projector tensor parameters.
    params : dict
        Dictionary containing the projector tensor parameters.
    
    Returns
    ----------
    new_proj_params : dict
        Dictionary containing the reconstructed projector tensor parameters.
    """
    new_proj_params = {}
    idx = 0
    for tid, ts_values in params.items():
        new_ts_values = {}
        for blk, data in ts_values.items():
            new_ts_values[blk] = vec_params[idx:idx+len(data.flatten())].reshape(data.shape)
            idx += len(data.flatten())
        new_proj_params[tid] = new_ts_values
    return new_proj_params




def decompose_permutation_into_transpositions(perm):
    """
    Decompose a permutation into a product of contiguous transpositions using bubble sort.
    
    Parameters
    ----------
    perm : list or array-like
        The permutation of integers.

    Returns
    -------
    transpositions : list of tuples
        A list of transpositions (i, i+1) applied to sort the permutation.
    """
    n = len(perm)
    transpositions = []
    
    # Make a copy of the permutation to sort
    perm_copy = list(perm)
    
    # Bubble sort with recording of transpositions
    for i in range(n):
        for j in range(n - 1):
            if perm_copy[j] > perm_copy[j + 1]:
                # Swap elements and record the transposition (j, j+1)
                perm_copy[j], perm_copy[j + 1] = perm_copy[j + 1], perm_copy[j]
                transpositions.append((j, j + 1))
    
    return tuple(transpositions)



# def argsort_jax(seq, reversed=False):
#     """Return the indices that would sort an array."""
#     return jnp.argsort(jnp.array(seq), descending=reversed)

def argsort(seq, reversed=False):
    """Return the indices that would sort an array."""
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reversed)

def calculate_perm_dict(ampctree, asc=True):
    """Calculate the permutation dictionary of the amplitude contraction tree."""
    if asc:
        return {i: tuple(argsort(tuple(left_tids)+tuple(right_tids))) for i, (_, left_tids, right_tids) in enumerate(ampctree.traverse())}
    else:
        return {i: tuple(argsort(   tuple(list(left_tids)[::-1])  +  tuple(list(right_tids)[::-1])   , reversed=True)) for i, (_, left_tids, right_tids) in enumerate(ampctree.traverse())}


def decompose_permutation_into_transpositions(perm, asc=True):
    """
    Decompose a permutation into a product of contiguous transpositions using bubble sort.
    
    Parameters
    ----------
    perm : list or array-like
        The permutation of integers.

    Returns
    -------
    transpositions : list of tuples
        A list of transpositions (i, i+1) applied to sort the permutation.
    """
    n = len(perm)
    transpositions = []
    
    # Make a copy of the permutation to sort
    perm_copy = list(perm)
    
    # Bubble sort with recording of transpositions
    for i in range(n):
        for j in range(n - 1):
            if perm_copy[j] > perm_copy[j + 1]:
                # Swap elements and record the transposition (j, j+1)
                perm_copy[j], perm_copy[j + 1] = perm_copy[j + 1], perm_copy[j]
                transpositions.append((j, j + 1))
    
    return tuple(transpositions) if asc else tuple(transpositions[::-1])



def calculate_phase_from_adjacent_trans_dict(ampctree, input_config, peps_parity, parities, adjacent_transposition_dict, adjacent_transposition_dict_desc):
    """
    parities: combined parities of config and PEPS
    input_config: input configuration parity
    peps_parity: PEPS parity
    """
    phase = 1

    # compute the phase from moving oddpos indices across odd-parity tensors during contraction
    for i, _ in adjacent_transposition_dict.items():
        gen_phase = 1
        tids, left_tids, right_tids = list(ampctree.traverse())[int(i)]
        left_parity = sum(parities[tid] for tid in left_tids) % 2
        right_parity = sum(parities[tid] for tid in right_tids) % 2
        gen_phase *= (-1)**(left_parity*right_parity)
        # print(i, left_tids, right_tids, gen_phase)
        phase *= gen_phase
        # print(phase)
    
    # compute the phase from moving the PEPS odd-parity tensor
    for i, transposition_list in adjacent_transposition_dict.items():
        gen_phase = 1
        _, left_tids, right_tids = list(ampctree.traverse())[int(i)]
        tids = tuple(left_tids)+tuple(right_tids)
        peps_parities_selected = [peps_parity[tid] for tid in tids]
        for transposition in transposition_list:
            gen_phase *= (-1)**(peps_parities_selected[transposition[0]]*peps_parities_selected[transposition[1]])
            peps_parities_selected[transposition[0]], peps_parities_selected[transposition[1]] = peps_parities_selected[transposition[1]], peps_parities_selected[transposition[0]]
        phase *= gen_phase
        # print(gen_phase)
    
    # compute the phase from moving the input configuration odd-parity tensor
    for i, transposition_list in adjacent_transposition_dict_desc.items():
        gen_phase = 1
        _, left_tids, right_tids = list(ampctree.traverse())[int(i)]
        tids = tuple(list(left_tids)[::-1])+tuple(list(right_tids)[::-1])
        input_parities_selected = [input_config[tid]%2 for tid in tids]
        for transposition in transposition_list:
            gen_phase *= (-1)**(input_parities_selected[transposition[0]]*input_parities_selected[transposition[1]])
            input_parities_selected[transposition[0]], input_parities_selected[transposition[1]] = input_parities_selected[transposition[1]], input_parities_selected[transposition[0]]
        phase *= gen_phase
        # print(gen_phase)

    # compute the phase from moving the PEPS odd-parity tensor across the accumulative input configuration odd-parity tensor during contraction
    gen_phase = 1
    for tids, left_tids, right_tids in list(ampctree.traverse()):
        left_parity = sum(peps_parity[tid] for tid in left_tids) % 2
        right_parity = sum(input_config[tid] for tid in right_tids) % 2
        gen_phase *= (-1)**(left_parity*right_parity)
    phase *= gen_phase
    # print(gen_phase)

    return phase


def check_phase(peps, peps_parity_config, state, adjacent_transposition_dict, adjacent_transposition_dict_desc, symmetry='Z2'):

    print(peps_parity_config)
    print(state)
    amp_test = (peps|product_bra_state(state, peps, symmetry).conj()).contract()

    amp_parity_config = (state+peps_parity_config)%2
    computed_phase = calculate_phase_from_adjacent_trans_dict(state, peps_parity_config, amp_parity_config, adjacent_transposition_dict,adjacent_transposition_dict_desc)
    true_phase = amp_test/peps.get_amp(state, conj=True,no_odd=True).contract()
    modified_amp = peps.get_amp(state, conj=True,no_odd=True).contract()*computed_phase

    print(
        f'Benchmark amplitude: {amp_test}\n', 
        f'No odd amplitude: {peps.get_amp(state, conj=False,no_odd=True).contract()}\n',
        f'No odd amplitude (reversed): {peps.get_amp(state, conj=False, no_odd=True, reverse=True).contract()}\n',  
        f'Computed phase: {computed_phase}\n',
        f'True phase: {true_phase}\n',
        f'Modified amplitude: {modified_amp}\n',
        f'Match? = {np.abs(modified_amp-amp_test)<1e-8}\n'
        )