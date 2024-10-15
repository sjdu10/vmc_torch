from autoray import numpy as np
import symmray as sr
import torch
import numpy
import quimb.tensor as qtn
import autoray as ar
from quimb.tensor.tensor_core import  *
from quimb.tensor.tensor_core import bonds, tags_to_oset, rand_uuid
from quimb.tensor.tensor_2d import Rotator2D, pairwise


# class fPEPS(qtn.PEPS):
#     def __init__(self, arrays, *, shape="urdlp", tags=None, site_ind_id="k{},{}", site_tag_id="I{},{}", x_tag_id="X{}", y_tag_id="Y{}", **tn_opts):
#         super().__init__(arrays, shape=shape, tags=tags, site_ind_id=site_ind_id, site_tag_id=site_tag_id, x_tag_id=x_tag_id, y_tag_id=y_tag_id, **tn_opts)
#         self.symmetry = self.arrays[0].symmetry
    
#     def product_bra_state(self, config, no_odd=False):
#         """Spinless fermion product bra state."""
#         product_tn = qtn.TensorNetwork()
#         backend = self.tensors[0].data.backend
#         for n, site in zip(config, self.sites):
#             p_ind = self.site_ind_id.format(*site)
#             p_tag = self.site_tag_id.format(*site)
#             tid = self.sites.index(site)
#             if no_odd:
#                 tsr_data = sr.FermionicArray.from_blocks(blocks={(int(n),):do('array', [1.0,], like=backend)}, duals=(True,), symmetry=self.symmetry, charge=0)
#             else:
#                 tsr_data = sr.FermionicArray.from_blocks(blocks={(int(n),):do('array', [1.0,], like=backend)}, duals=(True,),symmetry=self.symmetry, charge=int(n), oddpos=2*tid+1 if int(n)%2 else None)
#             tsr = qtn.Tensor(data=tsr_data, inds=(p_ind,),tags=(p_tag, 'bra'))
#             product_tn |= tsr
#         return product_tn
    
#     # NOTE: don't use @classmethod here, as we need to access the specific instance attributes
#     def get_amp(self, config, inplace=False, no_odd=False, conj=True, reverse=False):
#         """Get the amplitude of a configuration in a PEPS."""
#         peps = self if inplace else self.copy()

#         if no_odd:
#             # WARNING: if use oddpos is False, then the phase correction is not activated, so A*B might not be equal to B*A
#             # if parity(A) == parity(B) == 1, then A*B = -B*A
#             # However if one use oddpos to record the relative position of A and B, then A*B = B*A, the on-site contraction will be automatically phase corrected
#             for t in peps:
#                 t.modify(data = t.data.copy_with(charge=0, oddpos=()))

#         if conj:
#             if reverse:
#                 # not recommended
#                 amp = self.product_bra_state(config, no_odd).conj()|peps # <n|--->---T---, generate a local parity phase (which is just a global -1 for odd-parity n as |n> either has 0 or 1 parity)
#             else:
#                 amp = peps|self.product_bra_state(config, no_odd).conj()
#         else:
#             if reverse:
#                 # not recommended
#                 amp = self.product_bra_state(config, no_odd)|peps
#             else:
#                 amp = peps|self.product_bra_state(config, no_odd)
        
#         for site in peps.sites:
#             site_tag = peps.site_tag_id.format(*site)
#             amp.contract_(tags=site_tag)

#         amp.view_as_(
#             qtn.PEPS,
#             site_ind_id="k{},{}",
#             site_tag_id="I{},{}",
#             x_tag_id="X{}",
#             y_tag_id="Y{}",
#             Lx=peps.Lx,
#             Ly=peps.Ly,
#         )
#         return amp


# def generate_random_fpeps(Lx, Ly, D, seed, symmetry='Z2', Nf=0):
#     """Generate a random spinless fermionic square PEPS of shape (Lx, Ly)."""

#     assert symmetry == 'Z2' or symmetry == 'U1', "Only Z2 and U1 symmetries are supported."
    
#     edges = qtn.edges_2d_square(Lx, Ly)
#     site_info = sr.utils.parse_edges_to_site_info(
#         edges,
#         D,
#         phys_dim=2,
#         site_ind_id="k{},{}",
#         site_tag_id="I{},{}",
#     )

#     peps = qtn.TensorNetwork()
#     rng = np.random.default_rng(seed)
#     parity_config = np.zeros(Lx*Ly, dtype=int)

#     # generate a random binary string with Nf ones in it
#     if symmetry == 'U1':
#         parity_config = np.zeros(Lx*Ly, dtype=int)
#         parity_config[:Nf] = 1
#         rng.shuffle(parity_config)

#     for site, info in sorted(site_info.items()):
#         tid = site[0] * Ly + site[1]
#         # bond index charge distribution
#         block_indices = [
#             sr.BlockIndex({0: d // 2, 1: d // 2}, dual=dual)
#             for d, dual in zip(info["shape"][:-1], info["duals"][:-1])
#         ]
#         # physical index
#         p = info['shape'][-1]
#         block_indices.append(
#             sr.BlockIndex({0: p // 2, 1: p // 2}, dual=info["duals"][-1])
#         )

#         if symmetry == 'Z2':
#             data = sr.Z2FermionicArray.random(
#                 block_indices,
#                 charge=1 if parity_config[tid] else 0,
#                 seed=rng,
#                 oddpos=2*tid,
#             )
#         elif symmetry == 'U1':
#             data = sr.U1FermionicArray.random(
#                 block_indices,
#                 charge=1 if parity_config[tid] else 0,
#                 seed=rng,
#                 oddpos=2*tid,
#             )

#         peps |= qtn.Tensor(
#             data=data,
#             inds=info["inds"],
#             tags=info["tags"],
#         )

#     # required to view general TN as an actual PEPS
#     for i, j in site_info:
#         peps[f"I{i},{j}"].add_tag([f"X{i}", f"Y{j}"])

#     peps.view_as_(
#         fPEPS,
#         site_ind_id="k{},{}",
#         site_tag_id="I{},{}",
#         x_tag_id="X{}",
#         y_tag_id="Y{}",
#         Lx=Lx,
#         Ly=Ly,
#     )
#     peps = peps.copy() # set symmetry during initialization

#     return peps, parity_config

#------Amplitude Calculation------

class fPEPS(qtn.PEPS):
    def __init__(self, arrays, *, shape="urdlp", tags=None, site_ind_id="k{},{}", site_tag_id="I{},{}", x_tag_id="X{}", y_tag_id="Y{}", **tn_opts):
        super().__init__(arrays, shape=shape, tags=tags, site_ind_id=site_ind_id, site_tag_id=site_tag_id, x_tag_id=x_tag_id, y_tag_id=y_tag_id, **tn_opts)
        self.symmetry = self.arrays[0].symmetry
        self.spinless = True if self.phys_dim() == 2 else False
    
    def product_bra_state(self, config):
        product_tn = qtn.TensorNetwork()
        backend = self.tensors[0].data.backend
        dtype = eval(backend+'.'+self.tensors[0].data.dtype)

        if self.spinless:
            index_map = {0: 0, 1: 1}
            array_map = {
                0: do('array',[1.0,],like=backend,dtype=dtype), 
                1: do('array',[1.0,],like=backend,dtype=dtype)
            }
        else:
            if self.symmetry == 'Z2':
                index_map = {0:0, 1:1, 2:1, 3:0}
                array_map = {
                    0: do('array',[1.0, 0.0],like=backend,dtype=dtype), 
                    1: do('array',[1.0, 0.0],like=backend,dtype=dtype), 
                    2: do('array',[0.0, 1.0],like=backend,dtype=dtype), 
                    3: do('array',[0.0, 1.0],like=backend,dtype=dtype)
                }
            elif self.symmetry == 'U1':
                index_map = {0:0, 1:1, 2:1, 3:2}
                array_map = {
                    0: do('array',[1.0,],like=backend,dtype=dtype), 
                    1: do('array',[1.0, 0.0],like=backend,dtype=dtype), 
                    2: do('array',[0.0, 1.0],like=backend,dtype=dtype), 
                    3: do('array',[1.0,],like=backend,dtype=dtype)
                }

        for n, site in zip(config, self.sites):
            p_ind = self.site_ind_id.format(*site)
            p_tag = self.site_tag_id.format(*site)
            tid = self.sites.index(site)

            n_charge = index_map[int(n)]
            n_array = array_map[int(n)]

            oddpos = None
            if not self.spinless:
                assert self.symmetry == 'U1', "Only U1 symmetry is supported for spinful fermions for now."
                if int(n) == 1:
                    oddpos = (3*tid+1)*(-1)#**reverse
                elif int(n) == 2:
                    oddpos = (3*tid+2)*(-1)#**reverse
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
            tsr = qtn.Tensor(data=tsr_data, inds=(p_ind,),tags=(p_tag, 'bra'))
            product_tn |= tsr

        return product_tn
    
    # NOTE: don't use @classmethod here, as we need to access the specific instance attributes
    def get_amp(self, config, inplace=False, conj=True):
        """Get the amplitude of a configuration in a PEPS."""
        peps = self if inplace else self.copy()
        product_state = self.product_bra_state(config).conj() if conj else self.product_bra_state(config)
        
        amp = peps|product_state # ---T---<---|n>
        
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

def generate_random_fpeps(Lx, Ly, D, seed, symmetry='Z2', Nf=0, cyclic=False, spinless=True):
    """Generate a random spinless/spinful fermionic square PEPS of shape (Lx, Ly)."""

    assert symmetry == 'Z2' or symmetry == 'U1', "Only Z2 and U1 symmetries are supported."
    
    edges = qtn.edges_2d_square(Lx, Ly, cyclic=cyclic)
    site_info = sr.utils.parse_edges_to_site_info(
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

    for site, info in sorted(site_info.items()):
        tid = site[0] * Ly + site[1]
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
    product_tn = qtn.TensorNetwork()
    backend = psi.tensors[0].data.backend
    dtype = eval(backend+'.'+psi.tensors[0].data.dtype)
    if psi.spinless:
        index_map = {0: 0, 1: 1}
        array_map = {
            0: do('array',[1.0],like=backend,dtype=dtype), 
            1: do('array',[1.0],like=backend,dtype=dtype)
        }
    else:
        if psi.symmetry == 'Z2':
            index_map = {0:0, 1:1, 2:1, 3:0}
            array_map = {
                0: do('array',[1.0, 0.0],like=backend,dtype=dtype), 
                1: do('array',[1.0, 0.0],like=backend,dtype=dtype), 
                2: do('array',[0.0, 1.0],like=backend,dtype=dtype), 
                3: do('array',[0.0, 1.0],like=backend,dtype=dtype)
            }
        elif psi.symmetry == 'U1':
            index_map = {0:0, 1:1, 2:1, 3:2}
            array_map = {
                0: do('array',[1.0],like=backend,dtype=dtype), 
                1: do('array',[1.0, 0.0],like=backend,dtype=dtype), 
                2: do('array',[0.0, 1.0],like=backend,dtype=dtype), 
                3: do('array',[1.0],like=backend,dtype=dtype)
            }

    iter = zip(config, psi.sites) if not reverse else zip(config[::-1], psi.sites[::-1])

    for n, site in iter:
        p_ind = psi.site_ind_id.format(*site)
        p_tag = psi.site_tag_id.format(*site)
        tid = psi.sites.index(site)

        n_charge = index_map[int(n)]
        n_array = array_map[int(n)]

        oddpos = None
        if not psi.spinless:
            if int(n) == 1:
                oddpos = (3*tid+1)*(-1)
            elif int(n) == 2:
                oddpos = (3*tid+2)*(-1)
            elif int(n) == 3:
                # oddpos = ((3*tid+1)*(-1)**reverse, (3*tid+2)*(-1)**reverse)
                oddpos = None
        else:
            oddpos = (3*tid+1)*(-1)
        
        tsr_data = sr.FermionicArray.from_blocks(
            blocks={(n_charge,):n_array}, 
            duals=(dualness,),
            symmetry=psi.symmetry, 
            charge=n_charge, 
            oddpos=oddpos
        )
        
        if check:
            if int(n)==0:
                blocks = {(0,): do('array',[1.0],like=backend,dtype=dtype), (1,): do('array',[0.0, 0.0],like=backend,dtype=dtype), (2,):do('array',[0.0],like=backend,dtype=dtype)}
            elif int(n)==1:
                blocks = {(0,): do('array',[0.0],like=backend,dtype=dtype), (1,): do('array',[1.0, 0.0],like=backend,dtype=dtype), (2,):do('array',[0.0],like=backend,dtype=dtype)}
            elif int(n)==2:
                blocks = {(0,): do('array',[0.0],like=backend,dtype=dtype), (1,): do('array',[0.0, 1.0],like=backend,dtype=dtype), (2,):do('array',[0.0],like=backend,dtype=dtype)}
            elif int(n)==3:
                blocks = {(0,): do('array',[0.0],like=backend,dtype=dtype), (1,): do('array',[0.0, 0.0],like=backend,dtype=dtype), (2,):do('array',[1.0],like=backend,dtype=dtype)}
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

def get_amp(peps, config, inplace=False, symmetry='Z2', conj=True):
    """Get the amplitude of a configuration in a PEPS."""
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

# --- Utils for calculating global phase on product states ---

def get_spinful_parity_map():
    return {0:0, 1:1, 2:1, 3:0}

def get_spinful_charge_map():
    return {0:0, 1:1, 2:1, 3:2}

def from_spinful_ind_to_charge(config):
    charge_map = get_spinful_charge_map()
    return np.array([charge_map[n] for n in config])

def from_netket_config_to_quimb_config(netket_configs):
    def func(netket_config):
        """Translate netket spin-1/2 config to tensor network product state config"""
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

def detect_hopping(configi, configj):
    """Detect the hopping between two configurations"""
    site_ls = np.asarray(configi-configj).nonzero()[0]
    if len(site_ls) == 2:
        return site_ls
    else:
        return None

def calc_phase_netket(configi, configj):
    """Calculate the phase factor for the matrix element in the netket basis, where all spin-up spins are placed before all spin-down spins"""
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
    Globally, this convention gives a staggered spin configuration (|dududu...>) when the configuration is flattened to 1D."""
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

def insert_proj_peps(amp, max_bond, yrange, xrange=None, from_which='ymin'):
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