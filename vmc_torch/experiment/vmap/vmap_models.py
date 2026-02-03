import quimb as qu
import quimb.tensor as qtn
import torch
import torch.nn as nn
import math
from typing import Optional
from vmc_torch.nn_sublayers import SelfAttn_block_pos
from typing import List
from vmc_torch.experiment.vmap.vmap_torch_utils import use_jitter_svd
# ==============================================================================

class PEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        # register the flat list parameters
        self.params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ])

    
    def amplitude(self, x, params):
        tn = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()
    
    def vamp(self, x, params):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)

class PEPS_Model_reuse(nn.Module):
    def __init__(self, tn, max_bond, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.bMPS_x_skeletons = {}
        self.bMPS_y_skeletons = {}
        self.bMPS_params_x_in_dims = None
        self.bMPS_params_y_in_dims = None
        self.chi = max_bond

        # for torch, further flatten pytree into a single list
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        # register the flat list parameters
        self.params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ])
    
    def cache_bMPS_skeleton(self, x):
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        tn = qtn.unpack(params, self.skeleton)
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
        bMPS_params_dict = {}
        for key, tn in env_x.items():
            bMPS_params, skeleton = qtn.pack(tn)
            env_x[key] = skeleton
            bMPS_params_dict[key] = bMPS_params

        self.bMPS_x_skeletons = env_x
        bMPS_params_x_in_dims = qu.utils.tree_map(lambda _: 0, bMPS_params_dict)
        self.bMPS_params_x_in_dims = bMPS_params_x_in_dims

        env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
        bMPS_params_dict = {}
        for key, tn in env_y.items():
            bMPS_params, skeleton = qtn.pack(tn)
            env_y[key] = skeleton
            bMPS_params_dict[key] = bMPS_params
        self.bMPS_y_skeletons = env_y
        bMPS_params_y_in_dims = qu.utils.tree_map(lambda _: 0, bMPS_params_dict)
        self.bMPS_params_y_in_dims = bMPS_params_y_in_dims

    @use_jitter_svd()
    def cache_bMPS_params_vmap(self, x):
        # return a pytree (dict) of bMPS params for x and y environments
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        def cache_bMPS_params_single(x_single, params):
            tn = qtn.unpack(params, self.skeleton)
            amp = tn.isel({tn.site_ind(site): x_single[i] for i, site in enumerate(tn.sites)})
            env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
            bMPS_params_x_dict = {}
            for key, btn in env_x.items():
                bMPS_params = btn.get_params()
                bMPS_params_x_dict[key] = bMPS_params
            bMPS_params_y_dict = {}
            env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
            for key, btn in env_y.items():
                bMPS_params = btn.get_params()
                bMPS_params_y_dict[key] = bMPS_params
            return bMPS_params_x_dict, bMPS_params_y_dict
        return torch.vmap(
            cache_bMPS_params_single,
            in_dims=(0, None),
        )(x, params)
    
    @use_jitter_svd()
    def cache_bMPS_params_any_direction_vmap(self, x, direction='x'):
        # return a pytree (dict) of bMPS params for x or y environments
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        def cache_bMPS_params_x_single(x_single, params):
            tn = qtn.unpack(params, self.skeleton)
            amp = tn.isel({tn.site_ind(site): x_single[i] for i, site in enumerate(tn.sites)})
            env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
            amp_val = (env_x[('xmin', self.Lx//2)]|env_x[('xmax', self.Lx//2-1)]).contract()
            bMPS_params_x_dict = {}
            for key, btn in env_x.items():
                bMPS_params = btn.get_params()
                bMPS_params_x_dict[key] = bMPS_params
            return bMPS_params_x_dict, amp_val
        def cache_bMPS_params_y_single(x_single, params):
            tn = qtn.unpack(params, self.skeleton)
            amp = tn.isel({tn.site_ind(site): x_single[i] for i, site in enumerate(tn.sites)})
            env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
            amp_val = (env_y[('ymin', self.Ly//2)]|env_y[('ymax', self.Ly//2-1)]).contract()
            bMPS_params_y_dict = {}
            for key, btn in env_y.items():
                bMPS_params = btn.get_params()
                bMPS_params_y_dict[key] = bMPS_params
            return bMPS_params_y_dict, amp_val
        if direction == 'x':
            return torch.vmap(
                cache_bMPS_params_x_single,
                in_dims=(0, None),
            )(x, params)
        else:
            return torch.vmap(
                cache_bMPS_params_y_single,
                in_dims=(0, None),
            )(x, params)
    
    @use_jitter_svd()
    def update_bMPS_params_to_row_vmap(self, x, row_id, bMPS_params_x_batched, from_which='xmin'):
        # update the bMPS params to a specific row_id for all samples in the batch
        bMPS_key = (from_which, row_id)
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        def update_bMPS_params_x_single(x_single, params, row_id, bMPS_params_x, from_which):
            tn = qtn.unpack(params, self.skeleton)
            amp = tn.isel({tn.site_ind(site): x_single[i] for i, site in enumerate(tn.sites)})
            bMPS_to_row = qtn.unpack(bMPS_params_x[bMPS_key], self.bMPS_x_skeletons[bMPS_key])
            row_tn = amp.select([tn.row_tag(row_id)], which='any')
            # MPO-MPS two row TN
            updated_bMPS = (bMPS_to_row|row_tn)
            # contract to get the updated bMPS, row_id+1 for xmin, row_id-1 for xmax
            if from_which == 'xmin':
                if row_id == 0:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmin_(max_bond=self.chi, cutoff=0.0, xrange=[row_id-1, row_id])
                updated_bMPS_params = updated_bMPS.get_params()
                pytree_params, _ = qu.utils.tree_flatten(updated_bMPS_params, get_ref=True)
                _, pytree = qu.utils.tree_flatten(bMPS_params_x[(from_which, row_id+1)], get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(pytree_params, pytree)
                bMPS_params_x[(from_which, row_id+1)] = updated_bMPS_params # inplace update
            else:
                if row_id == amp.Ly-1:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmax_(max_bond=self.chi, cutoff=0.0, xrange=[row_id, row_id+1])
                updated_bMPS_params = updated_bMPS.get_params()
                pytree_params, _ = qu.utils.tree_flatten(updated_bMPS_params, get_ref=True)
                _, pytree = qu.utils.tree_flatten(bMPS_params_x[(from_which, row_id-1)], get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(pytree_params, pytree)
                bMPS_params_x[(from_which, row_id-1)] = updated_bMPS_params # inplace update
            return bMPS_params_x
        return torch.vmap(
            update_bMPS_params_x_single,
            in_dims=(0, None, None, self.bMPS_params_x_in_dims, None),
        )(x, params, row_id, bMPS_params_x_batched, from_which)
    
    @use_jitter_svd()
    def update_bMPS_params_to_col_vmap(self, x, col_id, bMPS_params_y_batched, from_which='ymin'):
        # update the bMPS params to a specific col_id for all samples in the batch
        bMPS_key = (from_which, col_id)
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        def update_bMPS_params_y_single(x_single, params, col_id, bMPS_params_y, from_which):
            tn = qtn.unpack(params, self.skeleton)
            amp = tn.isel({tn.site_ind(site): x_single[i] for i, site in enumerate(tn.sites)})
            bMPS_to_col = qtn.unpack(bMPS_params_y[bMPS_key], self.bMPS_y_skeletons[bMPS_key])
            col_tn = amp.select([tn.col_tag(col_id)], which='any')
            # MPO-MPS two col TN
            updated_bMPS = (bMPS_to_col|col_tn)
            # contract to get the updated bMPS, col_id+1 for ymin, col_id-1 for ymax
            if from_which == 'ymin':
                if col_id == 0:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[col_id-1, col_id])
                updated_bMPS_params = updated_bMPS.get_params()
                pytree_params, _ = qu.utils.tree_flatten(updated_bMPS_params, get_ref=True)
                _, pytree = qu.utils.tree_flatten(bMPS_params_y[(from_which, col_id+1)], get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(pytree_params, pytree)
                bMPS_params_y[(from_which, col_id+1)] = updated_bMPS_params # inplace update
            else:
                if col_id == amp.Lx-1:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[col_id, col_id+1])
                updated_bMPS_params = updated_bMPS.get_params()
                pytree_params, _ = qu.utils.tree_flatten(updated_bMPS_params, get_ref=True)
                _, pytree = qu.utils.tree_flatten(bMPS_params_y[(from_which, col_id-1)], get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(pytree_params, pytree)
                bMPS_params_y[(from_which, col_id-1)] = updated_bMPS_params # inplace update
            return bMPS_params_y
        return torch.vmap(
            update_bMPS_params_y_single,
            in_dims=(0, None, None, self.bMPS_params_y_in_dims, None),
        )(x, params, col_id, bMPS_params_y_batched, from_which)
            
        
    def amp_tn(self, x):
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        tn = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        return amp
    
    def amplitude(
        self,
        x,
        params,
        bMPS_keys=None,
        bMPS_params_xmin=None,
        bMPS_params_xmax=None,
        bMPS_params_ymin=None,
        bMPS_params_ymax=None,
        selected_rows=None,
        selected_cols=None,
    ):
        tn = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})

        # replace the x-environment with the cached one
        if bMPS_params_xmin is not None and bMPS_params_xmax is not None and bMPS_keys is not None:
            bMPS_min = qtn.unpack(bMPS_params_xmin, self.bMPS_x_skeletons[bMPS_keys[0]])
            bMPS_max = qtn.unpack(bMPS_params_xmax, self.bMPS_x_skeletons[bMPS_keys[1]])
            rows = amp.select([tn.row_tag(row) for row in selected_rows], which='any')
            amp_reuse = (bMPS_min|rows|bMPS_max)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id = tn._site_tag_id,
                x_tag_id = tn._x_tag_id,
                y_tag_id = tn._y_tag_id,
                Lx = tn._Lx,
                Ly = tn._Ly,
                site_ind_id = tn._site_ind_id,
            )
            if self.chi > 0:
                amp_reuse.contract_boundary_from_xmin_(max_bond=self.chi, cutoff=0.0, xrange=[bMPS_keys[0][1], bMPS_keys[1][1]+1])
            return amp_reuse.contract()
        # replace the y-environment with the cached one
        if bMPS_params_ymin is not None and bMPS_params_ymax is not None and bMPS_keys is not None:
            bMPS_min = qtn.unpack(bMPS_params_ymin, self.bMPS_y_skeletons[bMPS_keys[0]])
            bMPS_max = qtn.unpack(bMPS_params_ymax, self.bMPS_y_skeletons[bMPS_keys[1]])
            cols = amp.select([tn.col_tag(col) for col in selected_cols], which='any')
            amp_reuse = (bMPS_min|cols|bMPS_max)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id = tn._site_tag_id,
                x_tag_id = tn._x_tag_id,
                y_tag_id = tn._y_tag_id,
                Lx = tn._Lx,
                Ly = tn._Ly,
                site_ind_id = tn._site_ind_id,
            )
            if self.chi > 0:
                amp_reuse.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[bMPS_keys[0][1], bMPS_keys[1][1]+1])
            return amp_reuse.contract()

        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
            
        return amp.contract()
    
    @use_jitter_svd()
    def vamp(
        self,
        x,
        params,
        bMPS_keys=None,
        bMPS_params_xmin=None,
        bMPS_params_xmax=None,
        bMPS_params_ymin=None,
        bMPS_params_ymax=None,
        selected_rows=None,
        selected_cols=None,
    ):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        if bMPS_params_xmin is not None and bMPS_params_xmax is not None:
            return torch.vmap(
                self.amplitude,
                in_dims=(
                    0,
                    None,
                    None,
                    self.bMPS_params_x_in_dims[bMPS_keys[0]],
                    self.bMPS_params_x_in_dims[bMPS_keys[1]],
                    None,
                    None,
                    None,
                    None,
                ),
            )(x, params, bMPS_keys, bMPS_params_xmin, bMPS_params_xmax, bMPS_params_ymin, bMPS_params_ymax, selected_rows, selected_cols)
        
        if bMPS_params_ymin is not None and bMPS_params_ymax is not None:
            return torch.vmap(
                self.amplitude,
                in_dims=(
                    0,
                    None,
                    None,
                    None,
                    None,
                    self.bMPS_params_y_in_dims[bMPS_keys[0]],
                    self.bMPS_params_y_in_dims[bMPS_keys[1]],
                    None,
                    None,
                ),
            )(x, params, bMPS_keys, bMPS_params_xmin, bMPS_params_xmax, bMPS_params_ymin, bMPS_params_ymax, selected_rows, selected_cols)

        return torch.vmap(
            self.amplitude,
            in_dims=(0, None, None, None, None, None, None, None, None),
        )(
            x,
            params,
            bMPS_keys,
            bMPS_params_xmin,
            bMPS_params_xmax,
            bMPS_params_ymin,
            bMPS_params_ymax,
            selected_rows,
            selected_cols,
        )

    def forward(
        self,
        x,
        bMPS_params_x_batched=None,
        bMPS_params_y_batched=None,
        selected_rows=None,
        selected_cols=None,
    ):
        bMPS_params_xmin = None
        bMPS_params_xmax = None
        bMPS_params_ymin = None
        bMPS_params_ymax = None
        bMPS_keys = None

        if selected_rows is not None:
            bMPS_keys = [('xmin', min(selected_rows)), ('xmax', max(selected_rows))]
            bMPS_params_xmin = bMPS_params_x_batched[bMPS_keys[0]]
            bMPS_params_xmax = bMPS_params_x_batched[bMPS_keys[1]]
        if selected_cols is not None:
            bMPS_keys = [('ymin', min(selected_cols)), ('ymax', max(selected_cols))]
            bMPS_params_ymin = bMPS_params_y_batched[bMPS_keys[0]]
            bMPS_params_ymax = bMPS_params_y_batched[bMPS_keys[1]]
        
        return self.vamp(
            x,
            self.params,
            bMPS_keys=bMPS_keys,
            bMPS_params_xmin=bMPS_params_xmin,
            bMPS_params_xmax=bMPS_params_xmax,
            bMPS_params_ymin=bMPS_params_ymin,
            bMPS_params_ymax=bMPS_params_ymax,
            selected_rows=selected_rows,
            selected_cols=selected_cols,
        )

class fPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, dtype=torch.float64, compile=False, contract_boundary_opts={}, **kwargs):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.contract_boundary_opts = contract_boundary_opts
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        # register the flat list parameters
        self.params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ])

        self._vmapped_amplitude = torch.vmap(
            self.amplitude,
            in_dims=(0, None),
            randomness='different',
        ) # pre-vmap the amplitude function for efficiency

        if compile:
            self._vmapped_amplitude = torch.compile(
                self._vmapped_amplitude, 
                fullgraph=False,
                mode="default",
            )
    
    def amplitude(self, x, params):
        tn = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_xmin_(max_bond=self.chi, cutoff=0.0, xrange=[0, amp.Lx//2-1], **self.contract_boundary_opts)
            amp.contract_boundary_from_xmax_(max_bond=self.chi, cutoff=0.0, xrange=[amp.Lx//2, amp.Lx-1], **self.contract_boundary_opts)
        return amp.contract()
    
    def vamp(self, x, params):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return self._vmapped_amplitude(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)


# ============================= Helper functions for fPEPS with reusing bMPS envs =============================
def is_vmap_compatible(x):
    """
    Check if a node is compatible with vmap (Tensor or Number).
    """
    return isinstance(x, torch.Tensor)

def is_quimb_place_holder(x):
    return isinstance(x, qu.tensor.interface.Placeholder)

def _get_params_ftn_pytree(ftn):
    ftn_params_raw, _ = qtn.pack(ftn)
    ftn_params = {}

    for key in ftn_params_raw.keys():
        # 1. Convert to raw pytree (contains None, 'Z2', etc.)
        raw_tree = ftn.tensor_map[key].data.to_pytree()
        ftn_params[key] = raw_tree
    return ftn_params

def pack_ftn(ftn):
    # Get raw params and skeleton from Quimb
    ftn_params_raw, skeleton = qtn.pack(ftn)
    ftn_params = {}

    for key in ftn_params_raw.keys():
        # 1. Convert to raw pytree (contains None, 'Z2', etc.)
        raw_tree = ftn.tensor_map[key].data.to_pytree()
        ftn_params[key] = raw_tree
    flat_ftn_params, skeleton_tree = qu.utils.tree_flatten(ftn_params,
                                              get_ref=True,
                                              is_leaf=is_vmap_compatible)
    flat_ftn_params = qu.utils.tree_map(lambda x: torch.as_tensor(x), flat_ftn_params, is_leaf=lambda x: isinstance(x, bool))
    return flat_ftn_params, skeleton


def unpack_ftn(flat_ftn_params, skeleton):
    # Create a shallow copy of the skeleton to modify
    ftn = skeleton.copy()
    ftn_params = _get_params_ftn_pytree(ftn)
    _, pytree = qu.utils.tree_flatten(
        ftn_params,
        get_ref=True,
        is_leaf=lambda x: is_vmap_compatible(x) or is_quimb_place_holder(x),
    )
    ftn_params = qu.utils.tree_unflatten(flat_ftn_params, pytree)
    for key in ftn_params.keys():
        new_data = ftn.tensor_map[key].data.from_pytree(ftn_params[key])
        ftn.tensor_map[key].modify(data=new_data)

    return ftn

def get_params_ftn(ftn):
    flat_ftn_params, _ = pack_ftn(ftn)
    return flat_ftn_params
# =============================================================================================================
class fPEPS_Model_reuse(nn.Module):

    def __init__(self, tn, max_bond, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()

        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.bMPS_x_skeletons = {}
        self.bMPS_y_skeletons = {}
        self.bMPS_params_x_in_dims = None
        self.bMPS_params_y_in_dims = None
        self.chi = max_bond

        # for torch, further flatten pytree into a single list
        params_flat, params_pytree = qu.utils.tree_flatten(params,
                                                           get_ref=True)
        self.params_pytree = params_pytree

        # register the flat list parameters
        self.params = torch.nn.ParameterList(
            [torch.as_tensor(x, dtype=self.dtype) for x in params_flat])

    def cache_bMPS_skeleton(self, x):
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        tns = qtn.unpack(
            params, self.skeleton
        )  # when unpacking tns, use quimb native unpack function is enough
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })
        env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
        bMPS_params_dict = {}
        for key, tn in env_x.items():
            bMPS_params, skeleton = pack_ftn(tn)
            env_x[key] = skeleton
            bMPS_params_dict[key] = bMPS_params

        self.bMPS_x_skeletons = env_x
        bMPS_params_x_in_dims = qu.utils.tree_map(lambda _: 0,
                                                  bMPS_params_dict)
        self.bMPS_params_x_in_dims = bMPS_params_x_in_dims

        env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
        bMPS_params_dict = {}
        for key, tn in env_y.items():
            bMPS_params, skeleton = pack_ftn(tn)
            env_y[key] = skeleton
            bMPS_params_dict[key] = bMPS_params
        self.bMPS_y_skeletons = env_y
        bMPS_params_y_in_dims = qu.utils.tree_map(lambda _: 0,
                                                  bMPS_params_dict)
        self.bMPS_params_y_in_dims = bMPS_params_y_in_dims

    def cache_bMPS_params_vmap(self, x):
        # return a pytree (dict) of bMPS params for x and y environments
        # For fermions, need to record every tensor's fermionic info too.
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)

        def cache_bMPS_params_single(x_single, params):
            tns = qtn.unpack(
                params, self.skeleton
            )  # when unpacking tns, use quimb native unpack function is enough
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
            bMPS_params_x_dict = {}
            for key, btn in env_x.items():
                bMPS_params = get_params_ftn(btn)
                bMPS_params_x_dict[key] = bMPS_params
            bMPS_params_y_dict = {}
            env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
            for key, btn in env_y.items():
                bMPS_params = get_params_ftn(btn)
                bMPS_params_y_dict[key] = bMPS_params

            return bMPS_params_x_dict, bMPS_params_y_dict

        return torch.vmap(
            cache_bMPS_params_single,
            in_dims=(0, None),
        )(x, params)

    def cache_bMPS_params_any_direction_vmap(self, x, direction='x'):
        # return a pytree (dict) of bMPS params for x or y environments
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)

        def cache_bMPS_params_x_single(x_single, params):
            tns = qtn.unpack(
                params, self.skeleton
            )  # when unpacking tns, use quimb native unpack function is enough
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0)
            amp_val = (env_x[('xmin', self.Lx // 2)]
                       | env_x[('xmax', self.Lx // 2 - 1)]).contract()
            bMPS_params_x_dict = {}
            for key, btn in env_x.items():
                bMPS_params = get_params_ftn(btn)
                bMPS_params_x_dict[key] = bMPS_params
            return bMPS_params_x_dict, amp_val

        def cache_bMPS_params_y_single(x_single, params):
            tns = qtn.unpack(
                params, self.skeleton
            )  # when unpacking tns, use quimb native unpack function is enough
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            env_y = amp.compute_y_environments(max_bond=self.chi, cutoff=0.0)
            amp_val = (env_y[('ymin', self.Ly // 2)]
                       | env_y[('ymax', self.Ly // 2 - 1)]).contract()
            bMPS_params_y_dict = {}
            for key, btn in env_y.items():
                bMPS_params = get_params_ftn(btn)
                bMPS_params_y_dict[key] = bMPS_params
            return bMPS_params_y_dict, amp_val

        if direction == 'x':
            return torch.vmap(
                cache_bMPS_params_x_single,
                in_dims=(0, None),
            )(x, params)
        else:
            return torch.vmap(
                cache_bMPS_params_y_single,
                in_dims=(0, None),
            )(x, params)

    def update_bMPS_params_to_row_vmap(self,
                                       x,
                                       row_id,
                                       bMPS_params_x_batched,
                                       from_which='xmin'):
        # update the bMPS params to a specific row_id for all samples in the batch
        bMPS_key = (from_which, row_id)
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)

        def update_bMPS_params_x_single(x_single, params, row_id,
                                        bMPS_params_x, from_which):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            bMPS_to_row = unpack_ftn(bMPS_params_x[bMPS_key],
                                     self.bMPS_x_skeletons[bMPS_key])
            row_tn = amp.select([tns.row_tag(row_id)], which='any')
            # MPO-MPS two row TN
            updated_bMPS = (bMPS_to_row | row_tn)
            # contract to get the updated bMPS, row_id+1 for xmin, row_id-1 for xmax
            if from_which == 'xmin':
                if row_id == 0:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmin_(
                        max_bond=self.chi,
                        cutoff=0.0,
                        xrange=[row_id - 1, row_id])
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(updated_bMPS_params,
                                                         get_ref=True)
                _, pytree = qu.utils.tree_flatten(bMPS_params_x[(from_which,
                                                                 row_id + 1)],
                                                  get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree)
                bMPS_params_x[(from_which, row_id +
                               1)] = updated_bMPS_params  # inplace update
            else:
                if row_id == amp.Ly - 1:
                    updated_bMPS = row_tn
                else:
                    updated_bMPS.contract_boundary_from_xmax_(
                        max_bond=self.chi,
                        cutoff=0.0,
                        xrange=[row_id, row_id + 1])
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(updated_bMPS_params,
                                                         get_ref=True)
                _, pytree = qu.utils.tree_flatten(bMPS_params_x[(from_which,
                                                                 row_id - 1)],
                                                  get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree)
                bMPS_params_x[(from_which, row_id -
                               1)] = updated_bMPS_params  # inplace update
            return bMPS_params_x

        return torch.vmap(
            update_bMPS_params_x_single,
            in_dims=(0, None, None, self.bMPS_params_x_in_dims, None),
        )(x, params, row_id, bMPS_params_x_batched, from_which)

    def update_bMPS_params_to_col_vmap(self,
                                       x,
                                       col_id,
                                       bMPS_params_y_batched,
                                       from_which='ymin'):
        # update the bMPS params to a specific col_id for all samples in the batch
        bMPS_key = (from_which, col_id)
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)

        def update_bMPS_params_y_single(x_single, params, col_id,
                                        bMPS_params_y, from_which):
            tns = qtn.unpack(params, self.skeleton)
            amp = tns.isel({
                tns.site_ind(site): x_single[i]
                for i, site in enumerate(tns.sites)
            })
            bMPS_to_col = unpack_ftn(bMPS_params_y[bMPS_key],
                                     self.bMPS_y_skeletons[bMPS_key])
            col_tn = amp.select([tns.col_tag(col_id)], which='any')
            # MPO-MPS two col TN
            updated_bMPS = (bMPS_to_col | col_tn)
            # contract to get the updated bMPS, col_id+1 for ymin, col_id-1 for ymax
            if from_which == 'ymin':
                if col_id == 0:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymin_(
                        max_bond=self.chi,
                        cutoff=0.0,
                        yrange=[col_id - 1, col_id])
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(updated_bMPS_params,
                                                         get_ref=True)
                _, pytree = qu.utils.tree_flatten(bMPS_params_y[(from_which,
                                                                 col_id + 1)],
                                                  get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree)
                bMPS_params_y[(from_which, col_id +
                               1)] = updated_bMPS_params  # inplace update
            else:
                if col_id == amp.Lx - 1:
                    updated_bMPS = col_tn
                else:
                    updated_bMPS.contract_boundary_from_ymax_(
                        max_bond=self.chi,
                        cutoff=0.0,
                        yrange=[col_id, col_id + 1])
                updated_bMPS_params = get_params_ftn(updated_bMPS)
                pytree_params, _ = qu.utils.tree_flatten(updated_bMPS_params,
                                                         get_ref=True)
                _, pytree = qu.utils.tree_flatten(bMPS_params_y[(from_which,
                                                                 col_id - 1)],
                                                  get_ref=True)
                updated_bMPS_params = qu.utils.tree_unflatten(
                    pytree_params, pytree)
                bMPS_params_y[(from_which, col_id -
                               1)] = updated_bMPS_params  # inplace update
            return bMPS_params_y

        return torch.vmap(
            update_bMPS_params_y_single,
            in_dims=(0, None, None, self.bMPS_params_y_in_dims, None),
        )(x, params, col_id, bMPS_params_y_batched, from_which)

    def amp_tn(self, x):
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        tns = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })
        return amp

    def amplitude(
        self,
        x,
        params,
        bMPS_keys=None,
        bMPS_params_xmin=None,
        bMPS_params_xmax=None,
        bMPS_params_ymin=None,
        bMPS_params_ymax=None,
        selected_rows=None,
        selected_cols=None,
    ):
        tns = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })

        # replace the x-environment with the cached one
        if bMPS_params_xmin is not None and bMPS_params_xmax is not None and bMPS_keys is not None:
            bMPS_min = unpack_ftn(bMPS_params_xmin,
                                  self.bMPS_x_skeletons[bMPS_keys[0]])
            bMPS_max = unpack_ftn(bMPS_params_xmax,
                                  self.bMPS_x_skeletons[bMPS_keys[1]])
            rows = amp.select([tns.row_tag(row) for row in selected_rows],
                              which='any')
            amp_reuse = (bMPS_min | rows | bMPS_max)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id=tns._site_tag_id,
                x_tag_id=tns._x_tag_id,
                y_tag_id=tns._y_tag_id,
                Lx=tns._Lx,
                Ly=tns._Ly,
                site_ind_id=tns._site_ind_id,
            )
            # if self.chi > 0:
            #     amp_reuse.contract_boundary_from_xmin_(max_bond=self.chi, cutoff=0.0, xrange=[bMPS_keys[0][1], bMPS_keys[1][1]+1])
            return amp_reuse.contract()
        # replace the y-environment with the cached one
        if bMPS_params_ymin is not None and bMPS_params_ymax is not None and bMPS_keys is not None:
            bMPS_min = unpack_ftn(bMPS_params_ymin,
                                  self.bMPS_y_skeletons[bMPS_keys[0]])
            bMPS_max = unpack_ftn(bMPS_params_ymax,
                                  self.bMPS_y_skeletons[bMPS_keys[1]])
            cols = amp.select([tns.col_tag(col) for col in selected_cols],
                              which='any')
            amp_reuse = (bMPS_min | cols | bMPS_max)
            amp_reuse.view_as_(
                qtn.PEPS,
                site_tag_id=tns._site_tag_id,
                x_tag_id=tns._x_tag_id,
                y_tag_id=tns._y_tag_id,
                Lx=tns._Lx,
                Ly=tns._Ly,
                site_ind_id=tns._site_ind_id,
            )
            # if self.chi > 0:
            #     amp_reuse.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[bMPS_keys[0][1], bMPS_keys[1][1]+1])
            return amp_reuse.contract()

        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi,
                                             cutoff=0.0,
                                             yrange=[0, amp.Ly // 2 - 1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi,
                                             cutoff=0.0,
                                             yrange=[amp.Ly // 2, amp.Ly - 1])

        return amp.contract()

    def vamp(
        self,
        x,
        params,
        bMPS_keys=None,
        bMPS_params_xmin=None,
        bMPS_params_xmax=None,
        bMPS_params_ymin=None,
        bMPS_params_ymax=None,
        selected_rows=None,
        selected_cols=None,
    ):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        if bMPS_params_xmin is not None and bMPS_params_xmax is not None:
            return torch.vmap(
                self.amplitude,
                in_dims=(
                    0,
                    None,
                    None,
                    self.bMPS_params_x_in_dims[bMPS_keys[0]],
                    self.bMPS_params_x_in_dims[bMPS_keys[1]],
                    None,
                    None,
                    None,
                    None,
                ),
            )(x, params, bMPS_keys, bMPS_params_xmin, bMPS_params_xmax,
              bMPS_params_ymin, bMPS_params_ymax, selected_rows, selected_cols)

        if bMPS_params_ymin is not None and bMPS_params_ymax is not None:
            return torch.vmap(
                self.amplitude,
                in_dims=(
                    0,
                    None,
                    None,
                    None,
                    None,
                    self.bMPS_params_y_in_dims[bMPS_keys[0]],
                    self.bMPS_params_y_in_dims[bMPS_keys[1]],
                    None,
                    None,
                ),
            )(x, params, bMPS_keys, bMPS_params_xmin, bMPS_params_xmax,
              bMPS_params_ymin, bMPS_params_ymax, selected_rows, selected_cols)

        return torch.vmap(
            self.amplitude,
            in_dims=(0, None, None, None, None, None, None, None, None),
        )(
            x,
            params,
            bMPS_keys,
            bMPS_params_xmin,
            bMPS_params_xmax,
            bMPS_params_ymin,
            bMPS_params_ymax,
            selected_rows,
            selected_cols,
        )

    def forward(
        self,
        x,
        bMPS_params_x_batched=None,
        bMPS_params_y_batched=None,
        selected_rows=None,
        selected_cols=None,
    ):
        bMPS_params_xmin = None
        bMPS_params_xmax = None
        bMPS_params_ymin = None
        bMPS_params_ymax = None
        bMPS_keys = None

        if selected_rows is not None:
            bMPS_keys = [('xmin', min(selected_rows)),
                         ('xmax', max(selected_rows))]
            bMPS_params_xmin = bMPS_params_x_batched[bMPS_keys[0]]
            bMPS_params_xmax = bMPS_params_x_batched[bMPS_keys[1]]
        if selected_cols is not None:
            bMPS_keys = [('ymin', min(selected_cols)),
                         ('ymax', max(selected_cols))]
            bMPS_params_ymin = bMPS_params_y_batched[bMPS_keys[0]]
            bMPS_params_ymax = bMPS_params_y_batched[bMPS_keys[1]]

        return self.vamp(
            x,
            self.params,
            bMPS_keys=bMPS_keys,
            bMPS_params_xmin=bMPS_params_xmin,
            bMPS_params_xmax=bMPS_params_xmax,
            bMPS_params_ymin=bMPS_params_ymin,
            bMPS_params_ymax=bMPS_params_ymax,
            selected_rows=selected_rows,
            selected_cols=selected_cols,
        )


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, **mha_kwargs):
        super().__init__()
        # instantiate the real MHA
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, **mha_kwargs
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = False,
    ):
        # internally use x for (query, key, value)
        # x should be of shape (batch_size, seq_length, embed_dim)
        return self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
    
class TransformerLayer(nn.Module):
    """
    单个 Transformer 层 (Pre-Norm 结构)
    结构: Input -> LayerNorm -> SelfAttention -> Residual Add -> Output
    """
    def __init__(self, embed_dim, num_heads, dtype=torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.dtype = dtype
        
        # Pre-Norm 放在 Attention 之前
        self.norm = nn.LayerNorm(embed_dim, dtype=dtype)
        
        # 你的自定义 Attention 模块
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads
        )
        self.attn.to(dtype=dtype)

    def forward(self, x):
        # x: (Batch, L, D)
        
        # Pre-Norm: 先 Norm 再进 Attention
        residual = x
        x_norm = self.norm(x)
        
        # Attention
        attn_out, _ = self.attn(x_norm)
        
        # Residual Connection
        return residual + attn_out

class SelfAttn_block_pos_batched(nn.Module):
    """ 
    支持多层堆叠的 Self-attention block with positional encoding 
    Args:
        depth (int): Transformer block 的层数 (default: 1)
    """
    def __init__(
        self, n_site, num_classes, embed_dim, attention_heads, depth=1, dtype=torch.float32
    ):
        super(SelfAttn_block_pos_batched, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.n_site = n_site
        self.dtype = dtype
        self.depth = depth

        # --- Embedding 部分 (vmap 优化版) ---
        # 直接使用 3D Parameter, 避免 forward 里的 stack 操作
        # Shape: (Sites, Embed_Dim, Classes)
        self.pos_weights = nn.Parameter(torch.empty(n_site, embed_dim, num_classes, dtype=dtype))
        self.pos_biases = nn.Parameter(torch.empty(n_site, embed_dim, dtype=dtype))
        
        # 初始化 Embedding 参数
        nn.init.kaiming_uniform_(self.pos_weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pos_weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.pos_biases, -bound, bound)

        # --- Transformer Layers (支持 depth) ---
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim, 
                num_heads=attention_heads, 
                dtype=dtype
            )
            for _ in range(depth)
        ])
        
        # 可选: 最后的 Norm (常见于 Pre-Norm 架构的末尾)
        self.final_norm = nn.LayerNorm(embed_dim, dtype=dtype)

    def forward(self, input_seq):
        # input_seq: (Batch, L) or (L,) inside vmap
        
        # --- Step 1: Vmap-safe One-hot Encoding ---
        # 1. Prepare output container
        out_shape = input_seq.shape + (self.num_classes,)
        one_hot_encoded = torch.zeros(out_shape, device=input_seq.device, dtype=self.dtype)

        # 2. Prepare indices (avoid .long() inside vmap if possible, or do it safely)
        indices = input_seq.unsqueeze(-1)
        if indices.dtype != torch.int64:
             indices = indices.to(torch.int64)

        # 3. Scatter (Out-of-place for vmap safety)
        one_hot_encoded = one_hot_encoded.scatter(-1, indices, 1.0)

        # --- Step 2: Batched Position-wise Embedding ---
        # Contract: (Batch, L, C) * (L, D, C) -> (Batch, L, D)
        # Using '...lc' to handle both batched and unbatched inputs
        embedded = torch.einsum('...lc,ldc->...ld', one_hot_encoded, self.pos_weights) + self.pos_biases

        # --- Step 3: Stacked Transformer Blocks ---
        x = embedded
        
        for layer in self.layers:
            x = layer(x)
            
        # --- Step 4: Final Norm ---
        x = self.final_norm(x)

        return x


# ==============================================================================
# Helper Module: Backflow Modules
# ==============================================================================

class PointwiseBackflow(nn.Module):
    """
    一个替换全连接 MLP 的高效模块。
    它对每个 site 独立应用相同的 MLP (Shared Weights),
    然后根据每个 site 实际需要的 TN 参数量进行裁剪和拼接。
    """
    def __init__(self, n_sites, embed_dim, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.param_sizes = param_sizes  # list, e.g., [32, 32, 128, 128, ...]
        self.max_size = max(param_sizes) # e.g., 128
        
        # 这是一个 "Local" MLP, 作用于 (Batch, N, embed_dim)
        # 参数量只与 hidden_dim 和 max_size 有关, 与 n_sites 无关！
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, self.max_size, dtype=dtype)
        )

    def forward(self, x):
        # x shape: (Batch, n_sites, embed_dim)
        
        # 1. 生成最大可能需要的参数 (Batch, n_sites, max_size)
        raw_out = self.net(x) # Linear layer automatically applies to last dim
        
        # 2. 根据每个 site 实际需要的参数量进行裁剪并拼接
        # 这一步保证了输出形状严格等于 ftn_params_length
        parts = []
        for i, size in enumerate(self.param_sizes):
            # 取出第 i 个 site 的前 size 个参数
            parts.append(raw_out[:, i, :size])
            
        # 3. 拼接成一个大向量: (Batch, Total_TN_Params)
        return torch.cat(parts, dim=1)

class Conv2dBackflow(nn.Module):
    """
    2D 卷积 Backflow 模块。
    将 (Batch, N_sites, Embed) 还原为 (Batch, Embed, Lx, Ly) 的 2D 图像结构, 
    利用 Conv2d 提取近邻信息 (上下左右), 再投影回参数空间。
    """
    def __init__(self, lx, ly, kernel_size, embed_dim, hidden_dim, param_sizes, dtype, pbc=False):
        super().__init__()
        
        # 几何完整性检查
        n_sites = len(param_sizes)
        assert lx * ly == n_sites, f"Lattice dims ({lx}x{ly}) do not match n_sites ({n_sites})"
        
        self.lx = lx
        self.ly = ly
        self.param_sizes = param_sizes
        self.max_size = max(param_sizes)
        self.pbc = pbc  # 是否使用周期性边界条件
        
        # 定义 2D 卷积层
        # Kernel=3, Padding=1 保证输出几何尺寸不变 (Lx, Ly)
        padding_mode = 'circular' if pbc else 'zeros'
        
        self.net = nn.Sequential(
            # Layer 1: 混合邻居信息 (Spatial Mixing)
            nn.Conv2d(
                in_channels=embed_dim, 
                out_channels=hidden_dim, 
                kernel_size=kernel_size, 
                padding=kernel_size // 2, 
                padding_mode=padding_mode,
                dtype=dtype
            ),
            nn.GELU(),
            
            # Layer 2: 投影到参数维度 (Pointwise Projection)
            # Kernel=1 等价于 Pointwise Linear, 但在 (C, H, W) 格式下运算
            nn.Conv2d(
                in_channels=hidden_dim, 
                out_channels=self.max_size, 
                kernel_size=1, 
                dtype=dtype
            )
        )
    
    def initialize_output_scale(self, scale):
        # 这个类知道自己的最后一层在 self.net[-1]
        last_layer = self.net[-1]
        print(f" -> Init Conv2d output scale: {scale}")
        torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
        if last_layer.bias is not None:
            torch.nn.init.zeros_(last_layer.bias)

    def forward(self, x):
        # Input x: (Batch, N_sites, Embed_Dim)
        B, N, D = x.shape
        
        # 1. Reshape sequence to image: (B, N, D) -> (B, D, Lx, Ly)
        # PyTorch Conv2d 需要 (Batch, Channel, Height, Width)
        x_2d = x.view(B, self.lx, self.ly, D).permute(0, 3, 1, 2)
        
        # 2. Apply 2D Convolutions
        # Out shape: (Batch, Max_Size, Lx, Ly)
        out_2d = self.net(x_2d)
        
        # 3. Flatten back to sequence: (B, Max_Size, Lx, Ly) -> (Batch, N, Max_Size)
        # 必须先 permute 回 (B, Lx, Ly, Max_Size) 以保证 site 顺序一致
        raw_out = out_2d.permute(0, 2, 3, 1).contiguous().view(B, N, self.max_size)
        
        # 4. 根据每个 site 实际需要的参数量进行裁剪并拼接 (逻辑不变)
        parts = []
        for i, size in enumerate(self.param_sizes):
            # raw_out[:, i, :size] shape is (Batch, size)
            parts.append(raw_out[:, i, :size])
            
        # 5. Concatenate: (Batch, Total_TN_Params)
        return torch.cat(parts, dim=1)


class DilatedCNNBackflow(nn.Module):
    def __init__(self, lx, ly, embed_dim, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.lx = lx
        self.ly = ly
        self.max_size = max(param_sizes)
        self.param_sizes = param_sizes
        
        self.net = nn.Sequential(
            # Layer 1: local mixing
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1, dtype=dtype),
            nn.GELU(),
            
            # Layer 2: (Dilation=2), larger receptive field
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, dtype=dtype),
            nn.GELU(),
            
            # Layer 3: (Dilation=4) 
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4, dtype=dtype),
            nn.GELU(),
            
            # Layer 4: Project back to parameter space
            nn.Conv2d(hidden_dim, self.max_size, kernel_size=1, dtype=dtype)
        )

    def forward(self, x):
        # x: (B, N, D) -> (B, D, Lx, Ly)
        B, N, D = x.shape
        x_2d = x.view(B, self.lx, self.ly, D).permute(0, 3, 1, 2)
        
        out_2d = self.net(x_2d) # (B, Max_Params, Lx, Ly)
        
        # Flatten back
        raw_out = out_2d.permute(0, 2, 3, 1).contiguous().view(B, N, self.max_size)
        
        # Split and Cat (same as before)
        parts = [raw_out[:, i, :size] for i, size in enumerate(self.param_sizes)]
        return torch.cat(parts, dim=1)


class UNetBackflow(nn.Module):
    def __init__(self, lx, ly, embed_dim, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.lx, self.ly = lx, ly
        self.param_sizes = param_sizes
        self.max_size = max(param_sizes)
        
        # Encoder (下采样)
        self.enc1 = nn.Conv2d(embed_dim, hidden_dim, 3, padding=1, dtype=dtype)
        self.pool = nn.MaxPool2d(2) # 尺寸减半
        
        # Bottleneck (全局信息处理)
        # 此时图像尺寸很小, 卷积核可以轻易覆盖全局
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1, dtype=dtype),
            nn.GELU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, padding=1, dtype=dtype),
            nn.GELU()
        )
        
        # Decoder (上采样)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Conv2d(hidden_dim * 2 + hidden_dim, hidden_dim, 3, padding=1, dtype=dtype) # +hidden_dim for skip connection
        
        # Output Head
        self.head = nn.Conv2d(hidden_dim, self.max_size, 1, dtype=dtype)
        self.act = nn.GELU()

    def forward(self, x):
        B, N, D = x.shape
        x_img = x.view(B, self.lx, self.ly, D).permute(0, 3, 1, 2)
        
        # 1. Encode
        e1 = self.act(self.enc1(x_img)) # (B, H, Lx, Ly)
        p1 = self.pool(e1)              # (B, H, Lx/2, Ly/2)
        
        # 2. Bottleneck (Global Mixing)
        b = self.bottleneck(p1)         # (B, 2H, Lx/2, Ly/2)
        
        # 3. Decode
        u1 = self.up(b)                 # (B, 2H, Lx, Ly)
        
        # 处理尺寸不匹配 (如果 Lx, Ly 是奇数)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = torch.nn.functional.interpolate(u1, size=e1.shape[-2:])
            
        # Skip Connection: Concatenate Encoder feature
        cat = torch.cat([u1, e1], dim=1) # (B, 3H, Lx, Ly)
        
        d1 = self.act(self.dec1(cat))
        
        # 4. Output
        out_2d = self.head(d1)
        
        raw_out = out_2d.permute(0, 2, 3, 1).contiguous().view(B, N, self.max_size)
        parts = [raw_out[:, i, :size] for i, size in enumerate(self.param_sizes)]
        return torch.cat(parts, dim=1)


class FourierBackflow(nn.Module):
    def __init__(self, lx, ly, embed_dim, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.lx, self.ly = lx, ly
        self.param_sizes = param_sizes
        self.max_size = max(param_sizes)
        
        # 1. 预处理卷积
        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, 1, dtype=dtype)
        
        # 2. 频域权重 (复数)
        # 我们保留一半的频率模式 (RFFT)
        self.n_modes_x = lx // 2 + 1
        self.n_modes_y = ly // 2 + 1
        
        # Complex weights: (Hidden, Hidden, Modes_X, Modes_Y)
        scale = 1 / (hidden_dim * hidden_dim)
        self.weights = nn.Parameter(
            scale * torch.randn(hidden_dim, hidden_dim, self.n_modes_x, self.n_modes_y, 2, dtype=torch.float32) # FFT通常用float32
        )
        
        # 3. 后处理卷积
        self.conv2 = nn.Conv2d(hidden_dim, self.max_size, 1, dtype=dtype)
        self.act = nn.GELU()

    def complex_mul2d(self, input, weights):
        # (Batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (Batch, out_channel, x, y)
        # 手动实现复数乘法
        # input: (B, C, X, Y, 2)
        # weights: (C, C, X, Y, 2)
        return torch.view_as_complex(torch.stack([
            input.real * weights[..., 0] - input.imag * weights[..., 1],
            input.real * weights[..., 1] + input.imag * weights[..., 0]
        ], dim=-1))

    def forward(self, x):
        B, N, D = x.shape
        x_img = x.view(B, self.lx, self.ly, D).permute(0, 3, 1, 2) # (B, D, Lx, Ly)
        
        # 1. 映射到 Hidden Dim
        x_h = self.act(self.conv1(x_img)) # (B, H, Lx, Ly)
        
        # 2. Fourier Transform (RFFT)
        # 转为 float32 做 FFT 比较稳
        x_ft = torch.fft.rfft2(x_h.float(), norm='ortho') # (B, H, Lx, Ly/2+1)
        
        # 3. Frequency Mixing
        # 这里简化: 不做复杂的模式截断, 直接对全频段做 Pointwise Conv
        # weights: (H, H, Lx, Ly/2+1) (Complex)
        w_complex = torch.view_as_complex(self.weights)
        
        # Einstein Summation for batch matrix multiplication in freq domain
        # b: batch, i: input_channel, o: output_channel, x: mode_x, y: mode_y
        out_ft = torch.einsum("bixy,ioxy->boxy", x_ft, w_complex)
        
        # 4. Inverse Fourier Transform
        x_out = torch.fft.irfft2(out_ft, s=(self.lx, self.ly), norm='ortho')
        x_out = x_out.to(dtype=self.conv1.weight.dtype) # 转回 float64
        
        # 5. Output Project
        # Residual connection is very important for FNO
        out_2d = self.conv2(x_out + x_h) 
        
        raw_out = out_2d.permute(0, 2, 3, 1).contiguous().view(B, N, self.max_size)
        parts = [raw_out[:, i, :size] for i, size in enumerate(self.param_sizes)]
        return torch.cat(parts, dim=1)


class GlobalMLPBackflow(nn.Module):
    """
    Backflow 模块: Transformer -> Flatten -> N 个独立的 MLP
    
    逻辑: 
    1. Transformer 输出 (Batch, N_sites, Embed_Dim)
    2. Flatten 成 (Batch, N_sites * Embed_Dim) 的全局特征向量
    3. 对于 PEPS 中的每一个 Site i:
       使用一个独立的 MLP_i, 输入全局特征向量, 输出该 Site 的参数更新量 (Param_Size_i)
    4. 拼接所有输出, 得到 (Batch, Total_Params)
    """
    def __init__(self, attn_block, n_sites, embed_dim, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.attn = attn_block
        self.n_sites = n_sites
        self.embed_dim = embed_dim
        self.dtype = dtype
        
        # 计算 Flatten 后的全局特征维度
        self.global_feat_dim = n_sites * embed_dim
        
        # 为每个 Site 创建一个独立的 MLP
        # 注意: 每个 tensor 的参数量 (p_size) 可能不同
        self.site_mlps = nn.ModuleList()
        
        for p_size in param_sizes:
            mlp = nn.Sequential(
                # Layer 1: Global Feat -> Hidden
                nn.Linear(self.global_feat_dim, hidden_dim, dtype=dtype),
                nn.GELU(),
                # Layer 2: Hidden -> Local Tensor Params
                nn.Linear(hidden_dim, p_size, dtype=dtype)
            )
            self.site_mlps.append(mlp)

    def forward(self, x):
        # x: (Batch, N_sites)
        
        # 1. Attention Layers
        # feats: (Batch, N_sites, Embed_Dim)
        feats = self.attn(x)
        
        # 2. Flatten to Global Feature Vector
        # global_vec: (Batch, N_sites * Embed_Dim)
        B = x.shape[0]
        global_vec = feats.view(B, -1)
        
        # 3. Apply Independent MLPs for each site
        outputs = []
        for mlp in self.site_mlps:
            # mlp(global_vec): (Batch, param_size_i)
            outputs.append(mlp(global_vec))
            
        # 4. Concatenate all params: (Batch, Total_TN_Params)
        return torch.cat(outputs, dim=1)

class LocallyConnected2d(nn.Module):
    """
    实现不共享权重的 2D 卷积 (Locally Connected Layer)。
    每个空间位置 (x, y) 都有自己独立的卷积核。
    """
    def __init__(self, in_channels, out_channels, output_size, kernel_size, padding_mode='zeros', bias=True, dtype=torch.float32):
        super().__init__()
        self.output_size = output_size # (Lx, Ly)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding_mode = padding_mode
        self.dtype = dtype
        
        Lx, Ly = output_size
        N_sites = Lx * Ly
        
        # 权重形状: (N_sites, Out_Channels, In_Channels * K * K)
        # 每一个 site 都有一个 (Out, In_flat) 的矩阵
        self.weight_flat_dim = in_channels * kernel_size * kernel_size
        self.weight = nn.Parameter(
            torch.empty((N_sites, out_channels, self.weight_flat_dim), dtype=dtype)
        )
        
        if bias:
            self.bias = nn.Parameter(
                torch.empty((N_sites, out_channels), dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 初始化
        # 这里的 fan_in 计算的是单个 kernel 的输入维度
        fan_in = self.weight_flat_dim
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (Batch, In_Channels, Lx, Ly)
        B, C, Lx, Ly = x.shape
        
        # 1. 处理 Padding
        # Unfold 只支持 zero padding。如果是 circular, 我们需要手动 pad。
        padding_amt = self.kernel_size // 2
        
        if self.padding_mode == 'circular':
            # 左右上下各 pad
            x_padded = F.pad(x, (padding_amt, padding_amt, padding_amt, padding_amt), mode='circular')
            # 既然手动 pad 了, unfold 时 padding=0
            unfold_padding = 0
        else:
            x_padded = x
            unfold_padding = padding_amt

        # 2. Unfold (提取滑窗)
        # Input: (B, C, H_pad, W_pad)
        # Output: (B, C * K * K, L_out) where L_out = Lx * Ly
        # 这一步把每个像素周围的邻居都拉成了一根长条
        patches = F.unfold(
            x_padded, 
            kernel_size=self.kernel_size, 
            padding=unfold_padding
        ) # Shape: (B, Flat_In, N_sites)

        # 3. 本地矩阵乘法 (Site-wise Matrix Multiplication)
        # patches: (Batch, Flat_In, N_sites)
        # weight:  (N_sites, Out_Chan, Flat_In)
        # Target:  (Batch, Out_Chan, N_sites)
        
        # Einsum 解释:
        # b: Batch
        # i: Flat_In (Input features per patch)
        # n: N_sites (Spatial location)
        # o: Out_Channels
        
        out = torch.einsum('bin,noi->bon', patches, self.weight)
        
        if self.bias is not None:
            # bias: (N_sites, Out_Channels) -> (1, Out_Channels, N_sites)
            out = out + self.bias.permute(1, 0).unsqueeze(0)
        
        # 4. 恢复 2D 形状
        out = out.view(B, self.out_channels, Lx, Ly)
        
        return out

class CoordinateAwareBackflow(nn.Module):
    """
    [Shared Output + Deep MLP Version]
    Optimized Backflow for Open Boundary Conditions (OBC).
    
    Instead of a single projection layer, this uses a multi-layer MLP (Pointwise) 
    as the output head. This significantly enhances the model's ability to fit 
    non-linear boundary behaviors using coordinate information.

    Attributes:
        lx (int): Width of the lattice.
        ly (int): Height of the lattice.
        max_size (int): The maximum number of parameters required by any single tensor site.
        param_sizes (List[int]): A list containing the specific parameter count for each site.
    """

    def __init__(
        self, 
        lx: int, 
        ly: int, 
        kernel_size: int, 
        embed_dim: int, 
        hidden_dim: int, 
        param_sizes: List[int], 
        dtype: torch.dtype, 
        pbc: bool = False, 
        mlp_depth: int = 2
    ):
        """
        Initializes the CoordinateAwareBackflow_DeepMLP module.

        Args:
            lx (int): The width of the lattice (dimension in x-direction).
            ly (int): The height of the lattice (dimension in y-direction).
            kernel_size (int): The size of the convolutional kernel (e.g., 3 for 3x3).
            embed_dim (int): The dimension of the input embeddings (physical features).
            hidden_dim (int): The number of channels in the hidden layers.
            param_sizes (List[int]): A list where the i-th element represents the number 
                of variational parameters required for the i-th tensor site.
            dtype (torch.dtype): The data type for the parameters (e.g., torch.float64).
            pbc (bool, optional): Whether to use Periodic Boundary Conditions. 
                Defaults to False (OBC).
            mlp_depth (int, optional): The depth of the output pointwise MLP. 
                Defaults to 2 (1 hidden layer + 1 output layer).
        """
        super().__init__()
        self.lx = lx
        self.ly = ly
        self.param_sizes = param_sizes
        self.max_size = max(param_sizes)
        self.dtype = dtype
        
        # --- 1. Coordinate Embedding (Pre-calculated) ---
        # Generate normalized coordinate grid [-1, 1]
        # Shape: (1, 2, Lx, Ly) -> Channel 0 is X, Channel 1 is Y
        x_coords = torch.linspace(-1, 1, steps=lx)
        y_coords = torch.linspace(-1, 1, steps=ly)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('coord_grid', torch.stack([grid_x, grid_y], dim=0).unsqueeze(0))
        
        # --- 2. Shared Feature Extractor ---
        # Input Channel = embed_dim (physical features) + 2 (coordinate features)
        padding_mode = 'circular' if pbc else 'zeros'
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim + 2,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                padding_mode=padding_mode,
                dtype=dtype
            ),
            nn.GELU(),
            # Optional: Add more layers here to deepen feature extraction
        )
        
        # --- 3. Shared Deep MLP Output Projection (Core Modification) ---
        # We construct a Pointwise MLP (Sequence of 1x1 Convs).
        # Structure: Hidden -> [MLP Hidden -> GELU] * (Depth - 1) -> Output
        # Note: 1x1 Conv2d is mathematically equivalent to applying a shared Linear layer to every pixel.
        
        layers = []
        # Hidden Layers
        for _ in range(mlp_depth - 1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, dtype=dtype))
            layers.append(nn.GELU()) # Non-linearity is key for fitting complex boundary effects
            
        # Final Output Layer
        # Project to parameter space, no activation function here
        layers.append(nn.Conv2d(hidden_dim, self.max_size, kernel_size=1, dtype=dtype))
        
        self.output_mlp = nn.Sequential(*layers)

    def initialize_output_scale(self, scale: float):
        """
        Initialize the output layer weights to be small values for perturbative updates.

        IMPORTANT: We only initialize the *last layer* of the MLP to be near-zero.
        Hidden layers should use standard initialization (e.g., Kaiming) to maintain variance.

        Args:
            scale (float): The standard deviation for the normal distribution used 
                to initialize the last layer's weights.
        """
        # Get the last layer from Sequential
        last_layer = self.output_mlp[-1]
        
        print(f" -> [Init] CoordinateAwareBackflow_DeepMLP: Clamping last layer '{type(last_layer).__name__}' weights to scale {scale}")
        
        torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
        if last_layer.bias is not None:
            torch.nn.init.zeros_(last_layer.bias)
            
        # Optional: Explicitly initialize hidden layers with Kaiming/He initialization
        # (PyTorch default for Conv2d is usually adequate, but explicit is safer)
        for i in range(len(self.output_mlp) - 1):
            layer = self.output_mlp[i]
            if isinstance(layer, nn.Conv2d):
                # He initialization for layers followed by GELU/ReLU
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the backflow network.

        Args:
            x (torch.Tensor): Input embeddings or indices. 
                Shape: (Batch, N_sites, Embed_Dim)

        Returns:
            torch.Tensor: The flattened parameter updates concatenated for all sites.
                Shape: (Batch, Total_TN_Params)
        """
        # Input x: (Batch, N_sites, Embed_Dim)
        B, N, D = x.shape
        
        # 1. Reshape sequence to image & Permute to (B, C, H, W)
        x_img = x.view(B, self.lx, self.ly, D).permute(0, 3, 1, 2)
        
        # 2. Concatenate Coordinates
        # Expand coords to match batch size
        coords = self.coord_grid.expand(B, -1, -1, -1).to(x.dtype)
        x_input = torch.cat([x_img, coords], dim=1)
        
        # 3. Extract Features (Position Aware due to input coordinates)
        features = self.feature_extractor(x_input)
        
        # 4. Deep MLP Projection (Pointwise)
        # Input: (B, Hidden, Lx, Ly) -> Output: (B, Max_Size, Lx, Ly)
        out_2d = self.output_mlp(features)
        
        # 5. Flatten back to sequence: (B, Max_Size, Lx, Ly) -> (B, N, Max_Size)
        raw_out = out_2d.flatten(2).permute(0, 2, 1)
        
        # 6. Clip and Concatenate according to param_sizes
        parts = []
        for i, size in enumerate(self.param_sizes):
            parts.append(raw_out[:, i, :size])
            
        return torch.cat(parts, dim=1)


class LocalSiteNetwork(nn.Module):
    """
    针对单个 Site 的独立网络：
    Input (Neighbor Indices) -> Embedding -> Self Attention -> MLP -> Param Delta
    """
    def __init__(self, n_neighbors, num_classes, embed_dim, attention_heads, hidden_dim, output_dim, dtype):
        super().__init__()
        self.dtype = dtype
        
        # 1. Independent Embedding for this site (or shared, but independent is more expressive)
        self.embedding = nn.Embedding(num_classes, embed_dim)
        
        # 2. Local Self Attention
        # Input: (Batch, n_neighbors, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            batch_first=True,
            dtype=dtype
        )
        
        # 3. Output MLP
        # Flatten input: n_neighbors * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(n_neighbors * embed_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, dtype=dtype)
        )

    def forward(self, x_local_indices):
        # x_local_indices: (Batch, n_neighbors) int64
        
        # Embed: (Batch, n_neighbors, D)
        h = self.embedding(x_local_indices).to(self.dtype)
        
        # Self Attention
        # attn_output: (Batch, n_neighbors, D)
        h_attn, _ = self.attn(h, h, h)
        
        # Residual + Norm (Optional, simplified here to just Attn output)
        h = h + h_attn
        
        # Flatten
        B = h.shape[0]
        h_flat = h.reshape(B, -1)
        
        # Project to Tensor Params
        return self.mlp(h_flat)

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

class LocalClusterBackflow(nn.Module):
    def __init__(self, lx, ly, radius, phys_dim, embed_dim, attn_heads, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.lx = lx
        self.ly = ly
        
        # 1. 计算 Receptive Fields
        # 假设 site index map 是 row-major (i * Ly + j)
        rf_dict = get_receptive_field_2d(lx, ly, radius)
        
        # 预存 Indices 以加速 forward (避免 list 查找)
        # self.rf_indices[i] 是第 i 个 site 的邻居列表
        n_sites = lx * ly
        self.rf_indices = []
        for i in range(n_sites):
            self.rf_indices.append(rf_dict[i])
            
        # 2. 创建 N 个独立的网络
        self.site_networks = nn.ModuleList()
        for i in range(n_sites):
            n_neighbors = len(self.rf_indices[i])
            output_dim = param_sizes[i]
            
            net = LocalSiteNetwork(
                n_neighbors=n_neighbors,
                num_classes=phys_dim,
                embed_dim=embed_dim,
                attention_heads=attn_heads,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dtype=dtype
            )
            self.site_networks.append(net)

    def initialize_output_scale(self, scale):
        print(f" -> [Init] LocalClusterBackflow: Clamping output weights to scale {scale}")
        for net in self.site_networks:
            # The last layer of the MLP is at index 2
            last_layer = net.mlp[-1]
            torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
            if last_layer.bias is not None:
                torch.nn.init.zeros_(last_layer.bias)

    def forward(self, x):
        # x: (Batch, N_sites) Int/Long
        outputs = []
        
        # 既然是 Independent Weights，循环是不可避免的
        # 但每个循环内部是 Batched 的
        for i, net in enumerate(self.site_networks):
            neighbors = self.rf_indices[i]
            
            # Gather local config: (Batch, n_neighbors)
            x_local = x[:, neighbors]
            
            # Run local net
            out_i = net(x_local)
            outputs.append(out_i)
            
        # Concatenate all params: (Batch, Total_Params)
        return torch.cat(outputs, dim=1)


# ==============================================================================
# Main Transformer based NN-fPEPS Model
# ==============================================================================

class BasefPEPSBackflowModel(nn.Module):
    """
    Base Class: Handles fPEPS parameter management, vmap contraction logic, 
    and general forward pass flow.
    
    Subclasses must define `self.nn_backflow` in `__init__` and call 
    `self.finish_initialization()`.
    """
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        dtype=torch.float64,
        debug_file=None,
        contract_boundary_opts={}
    ):
        super().__init__()
        self.contract_boundary_opts = contract_boundary_opts
        self.dtype = dtype
        self.debug_file = debug_file
        self.chi = max_bond
        self.nn_eta = nn_eta

        # --- 1. Tensor Network Setup (Common) ---
        # Extract raw arrays and skeleton
        params, skeleton = qtn.pack(tn)
        self.skeleton = skeleton
        self.skeleton.exponent = 0 
        
        # Flatten TN parameters into a single list and a PyTree structure
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        self.ftn_params_pytree = ftn_params_pytree

        # Register TN parameters as a ParameterList so optimizer can see them
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        
        # Metadata for reconstruction inside vmap
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)

        # Placeholders for Child class to fill
        self.nn_backflow = None 
        self.nn_backflow_generator = None 
        self.nn_param_names = None
        self.params = None

        # vamp func
        self._vamp = torch.vmap(self.tn_contraction, in_dims=(0, None, 0), randomness='different')

    def finish_initialization(self, init_scale=1e-5):
        """
        Must be called by the subclass after `self.nn_backflow` is defined.
        It registers all parameters and initializes weights.
        """
        if self.nn_backflow is None:
            raise ValueError("Child class must define self.nn_backflow before calling finish_initialization")

        # 1. Register NN parameter names for functional_call
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # 2. Combine all params into one list for the optimizer
        # Order: [TN Params ... NN Params]
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
        # 3. Initialize perturbation weights to be small
        self._init_weights_for_perturbation(scale=init_scale)

    def _init_weights_for_perturbation(self, scale=1e-5):
        """
        Delegates initialization to the backflow generator (Generic Interface Pattern).
        """
        target_module = self.nn_backflow_generator if self.nn_backflow_generator else self.nn_backflow
        
        if hasattr(target_module, 'initialize_output_scale'):
            target_module.initialize_output_scale(scale)
        else:
            # Fallback for simple structures
            print(f"Warning: {type(target_module).__name__} does not implement 'initialize_output_scale'.")
            # Try to guess standard layers (Sequential/ModuleList)
            last_layer = None
            if isinstance(target_module, nn.Sequential):
                last_layer = target_module[-1]
            
            if last_layer and hasattr(last_layer, 'weight'):
                 print(f" -> Initializing last layer {type(last_layer).__name__} with scale {scale}")
                 torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
                 if last_layer.bias is not None: 
                     torch.nn.init.zeros_(last_layer.bias)

    def tn_contraction(self, x, ftn_params, nn_output):
        """ 
        Core logic for vmap:
        1. Reconstruct TN parameters from vector.
        2. Add NN backflow correction.
        3. Pack into Quimb TN.
        4. Perform contraction.
        """
        # 1. Reconstruct the vector
        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        
        # 2. Add backflow (NN correction)
        # nn_output is a single sample correction vector
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        
        # 3. Unpack to PyTree (list of tensors)
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            nnftn_params.append(nnftn_params_vector[pointer:pointer+length].view(shape))
            pointer += length
        
        # Restore dictionary structure and unpack to Quimb TN
        nnftn_params = qu.utils.tree_unflatten(nnftn_params, self.ftn_params_pytree)
        tn = qtn.unpack(nnftn_params, self.skeleton)
        
        # 4. Contraction
        # Note: x here is a single sample (tensor of indices)
        # Select tensors based on configuration x
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        
        # Contract boundary environments if max_bond is set
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1], **self.contract_boundary_opts)
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1], **self.contract_boundary_opts)
        
        return amp.contract()

    def vamp(self, x, params):
        """
        Batched computation:
        1. functional_call to compute Backflow (Batch Mode).
        2. vmap to compute TN Contraction.
        """
        # 1. Split params into TN part and NN part
        n_ftn = len(self.ftn_params)
        ftn_params = params[:n_ftn]
        nn_params = params[n_ftn:]
        
        # Reconstruct NN param dict for functional_call
        nn_params_dict = dict(zip(self.nn_param_names, nn_params))

        # 2. Compute Backflow (Batch Mode)
        # nn_backflow handles the logic (Global Attn, Local Conv, or Independent Cluster)
        batch_nn_outputs = torch.func.functional_call(self.nn_backflow, nn_params_dict, x)

        # 3. vmap TN Contraction
        # Map over x (dim 0) and nn_outputs (dim 0)
        # We do NOT map over ftn_params (None)
        amps = self._vamp(x, ftn_params, batch_nn_outputs)
            
        return amps

    def forward(self, x):
        # Ensure inputs are long type for embeddings/indexing
        if x.dtype != torch.long:
             x = x.to(torch.long)
        
        # Forward pass wraps vamp with optional jitter context
        return self.vamp(x, self.params)
            
class Transformer_fPEPS_Model_Conv2d(BasefPEPSBackflowModel):
    """
    Subclass A: Global Attention + Convolutional Backflow.
    Suitable for systems with translational symmetry or requiring global context.
    """
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        attn_heads,
        attn_depth=1,
        conv_kernel_size=3,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        uniform_kernel=False,
        **kwargs,
    ):
        # 1. Call Base Init
        super().__init__(tn, max_bond, nn_eta, dtype, kwargs.get('debug_file'), contract_boundary_opts=kwargs.get('contract_boundary_opts', {}))

        # 2. Define NN Architecture
        # Part A: Global Position-aware Attention
        self.attn = SelfAttn_block_pos_batched(
            n_site=len(tn.sites),
            num_classes=tn.phys_dim(),
            embed_dim=embed_dim,
            attention_heads=attn_heads,
            depth=attn_depth,
            dtype=self.dtype,
        )
        
        # Part B: Convolutional Generator
        if uniform_kernel:
            # Assumes Conv2dBackflow is defined
            self.nn_backflow_generator = Conv2dBackflow(
                lx=tn.Lx, ly=tn.Ly, kernel_size=conv_kernel_size,
                embed_dim=embed_dim, hidden_dim=nn_hidden_dim,
                param_sizes=self.ftn_params_sizes, dtype=self.dtype
            )
        else:
            # Assumes CoordinateAwareBackflow is defined (Hybrid strategy)
            self.nn_backflow_generator = CoordinateAwareBackflow(
                lx=tn.Lx, ly=tn.Ly, kernel_size=conv_kernel_size,
                embed_dim=embed_dim, hidden_dim=nn_hidden_dim,
                param_sizes=self.ftn_params_sizes, dtype=self.dtype
            )
            
        # Combine into a sequential module
        self.nn_backflow = nn.Sequential(self.attn, self.nn_backflow_generator)

        # 3. Finalize initialization (Register params and init weights)
        self.finish_initialization(init_perturbation_scale)


class Transformer_fPEPS_Model_Cluster(BasefPEPSBackflowModel):
    """
    Subclass B: Local Cluster Backflow.
    Totally independent neural networks per site. No global attention.
    Input for each site is determined strictly by its receptive field.
    """
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        attn_heads,
        radius=1,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs,
    ):
        # 1. Call Base Init
        super().__init__(tn, max_bond, nn_eta, dtype, kwargs.get('debug_file'), contract_boundary_opts=kwargs.get('contract_boundary_opts', {}))
        
        # 2. Define NN Architecture (Local & Independent)
        # Assumes LocalClusterBackflow is defined
        self.nn_backflow_generator = LocalClusterBackflow(
            lx=tn.Lx,
            ly=tn.Ly,
            radius=radius,
            phys_dim=tn.phys_dim(),
            embed_dim=embed_dim,
            attn_heads=attn_heads,
            hidden_dim=nn_hidden_dim,
            param_sizes=self.ftn_params_sizes,
            dtype=self.dtype
        )
        
        # Direct assignment (no global attn prepended)
        self.nn_backflow = self.nn_backflow_generator

        # 3. Finalize initialization
        self.finish_initialization(init_perturbation_scale)
        

class Transformer_fPEPS_Model_DConv2d(nn.Module):
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        attn_heads,
        attn_depth=1,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs
    ):
        super().__init__()
        
        # 1. PEPS / Tensor Network Setup (Same as before)
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        self.ftn_params_pytree = ftn_params_pytree

        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        
        # Number of parameters info per tensor
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)
        
        # 2. Neural Network Setup
        self.nn_hidden_dim = nn_hidden_dim
        self.embed_dim = embed_dim
        self.attn_heads = attn_heads
        
        # 核心 Attention 模块
        self.attn_block = SelfAttn_block_pos_batched(
            n_site=len(tn.sites),
            num_classes=tn.phys_dim(),
            embed_dim=self.embed_dim,
            attention_heads=self.attn_heads,
            depth=attn_depth,
            dtype=self.dtype,
        )
        
        # PointwiseBackflow 会自动处理不同大小的 tensor
        self.nn_backflow_generator = DilatedCNNBackflow(
            lx=self.skeleton.Lx,
            ly=self.skeleton.Ly,
            embed_dim=self.embed_dim,
            hidden_dim=self.nn_hidden_dim,
            param_sizes=self.ftn_params_sizes,
            dtype=self.dtype
        )
        # combine attn and mlp into a single nn_backflow
        self.nn_backflow = nn.Sequential(
            self.attn_block,
            self.nn_backflow_generator
        )

        self.nn_eta = nn_eta
        
        # 注册参数名字以便 functional_call 使用
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # 将所有参数合并到一个 ParameterList
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
        # 初始化微扰
        self._init_weights_for_perturbation(scale=init_perturbation_scale)
    
    def _init_weights_for_perturbation(self, scale=1e-5):
        """
        Initialize the final output layer of the backflow network to small random values,
        """
        backflow_module = self.nn_backflow_generator
        output_layer = backflow_module.net[-1]
        # 兼容 Linear 和 Conv2d
        if isinstance(output_layer, (torch.nn.Linear, torch.nn.Conv2d)):
            print(f" -> Clamping output layer ({type(output_layer).__name__}) weights to scale {scale}")
            torch.nn.init.normal_(output_layer.weight, mean=0.0, std=scale)
            if output_layer.bias is not None:
                torch.nn.init.zeros_(output_layer.bias)
        # Optional
        for m in self.attn_block.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def tn_contraction(self, x, ftn_params, nn_output):
        """ 
        vmap tn contraction for a single sample
        """
        # 1. Reconstruct the vector
        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        # 2. Add backflow
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        
        # 3. Unpack and contract
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            nnftn_params.append(nnftn_params_vector[pointer:pointer+length].view(shape))
            pointer += length
        
        nnftn_params = qu.utils.tree_unflatten(nnftn_params, self.ftn_params_pytree)
        tn = qtn.unpack(nnftn_params, self.skeleton)
        
        # Site indexing and contraction
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()

    def vamp(self, x, params):
        # 1. Split params (TN vs NN)
        ftn_params = params[:len(self.ftn_params)]
        nn_params = params[len(self.ftn_params):]
        nn_params_dict = dict(zip(self.nn_param_names, nn_params))

        # 2. Compute Backflow for the WHOLE BATCH at once
        # 这会调用 DConv2dBackflow.forward
        # 输出 Shape: (Batch, Total_Params)
        batch_nn_outputs = torch.func.functional_call(self.nn_backflow, nn_params_dict, x)

        # 3. vmap TN contraction
        amps = torch.vmap(
            self.tn_contraction,
            in_dims=(0, None, 0),
        )(x, ftn_params, batch_nn_outputs)
        
        return amps

    def forward(self, x):
        return self.vamp(x, self.params)
                

class Transformer_fPEPS_Model_UNet(nn.Module):
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        attn_heads,
        attn_depth=1,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs,
    ):
        super().__init__()
        
        # === 1. TN Setup (Standard) ===
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        self.ftn_params_pytree = ftn_params_pytree
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)
        
        # === 2. NN Setup ===
        self.nn_hidden_dim = nn_hidden_dim
        self.embed_dim = embed_dim
        
        # Attention Block
        self.attn_block = SelfAttn_block_pos_batched(
            n_site=len(tn.sites),
            num_classes=tn.phys_dim(),
            embed_dim=self.embed_dim,
            attention_heads=attn_heads,
            depth=attn_depth,
            dtype=self.dtype,
        )
        
        # --- Backflow Generator: UNet ---
        self.nn_backflow_generator = UNetBackflow(
            lx=self.skeleton.Lx,
            ly=self.skeleton.Ly,
            embed_dim=self.embed_dim,
            hidden_dim=self.nn_hidden_dim,
            param_sizes=self.ftn_params_sizes,
            dtype=self.dtype
        )
        
        # Combine
        self.nn_backflow = nn.Sequential(
            self.attn_block,
            self.nn_backflow_generator
        )

        self.nn_eta = nn_eta
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
        self._init_weights_for_perturbation(scale=init_perturbation_scale)
    
    def _init_weights_for_perturbation(self, scale=1e-5):
        # UNet 的输出层通常命名为 head 或类似名字, 这里我们需要定位到 generator 的最后一层
        backflow_module = self.nn_backflow_generator
        # 假设 UNetBackflow 的最后一层叫做 head
        output_layer = backflow_module.head 
        
        if isinstance(output_layer, (torch.nn.Linear, torch.nn.Conv2d)):
            print(f" -> [UNet] Clamping output layer weights to scale {scale}")
            torch.nn.init.normal_(output_layer.weight, mean=0.0, std=scale)
            if output_layer.bias is not None:
                torch.nn.init.zeros_(output_layer.bias)
                
        # Init attention
        for m in self.attn_block.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    # ... (tn_contraction, vamp, forward are reused from previous models) ...
    tn_contraction = Transformer_fPEPS_Model_DConv2d.tn_contraction
    vamp = Transformer_fPEPS_Model_DConv2d.vamp
    forward = Transformer_fPEPS_Model_DConv2d.forward

class Transformer_fPEPS_Model_GlobalMLP(nn.Module):
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        attn_heads,
        attn_depth=1,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs,
    ):
        super().__init__()
        
        # 1. PEPS / Tensor Network Setup (Same as before)
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        self.ftn_params_pytree = ftn_params_pytree

        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        
        # 计算每个 Tensor 的参数量, 这对于 Global MLP 至关重要
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)
        
        # 2. Neural Network Setup
        self.nn_hidden_dim = nn_hidden_dim
        self.embed_dim = embed_dim
        self.attn_heads = attn_heads
        
        # 核心 Attention 模块
        self.attn_block = SelfAttn_block_pos_batched(
            n_site=len(tn.sites),
            num_classes=tn.phys_dim(),
            embed_dim=self.embed_dim,
            attention_heads=self.attn_heads,
            depth=attn_depth,
            dtype=self.dtype,
        )
        
        # --- NEW: 使用 GlobalMLPBackflow ---
        self.nn_backflow = GlobalMLPBackflow(
            attn_block=self.attn_block,
            n_sites=len(tn.sites),
            embed_dim=self.embed_dim,
            hidden_dim=self.nn_hidden_dim,
            param_sizes=self.ftn_params_sizes, # 传入每个 tensor 的具体大小
            dtype=self.dtype
        )

        self.nn_eta = nn_eta
        
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
        self._init_weights_for_perturbation(scale=init_perturbation_scale)
    
    def _init_weights_for_perturbation(self, scale=1e-5):
        """
        初始化每个 MLP 的最后一层（输出层）为接近 0 的随机值。
        """
        print(f"Initializing {len(self.nn_backflow.site_mlps)} site MLPs perturbation...")
        
        for i, mlp in enumerate(self.nn_backflow.site_mlps):
            # mlp 是一个 Sequential: [Linear -> GELU -> Linear(Output)]
            output_layer = mlp[-1] 
            
            if isinstance(output_layer, torch.nn.Linear):
                torch.nn.init.normal_(output_layer.weight, mean=0.0, std=scale)
                if output_layer.bias is not None:
                    torch.nn.init.zeros_(output_layer.bias)
            
            # (Optional) Initialize the first layer as well
            first_layer = mlp[0]
            if isinstance(first_layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(first_layer.weight)
                if first_layer.bias is not None:
                    torch.nn.init.zeros_(first_layer.bias)

    def _get_name(self):
        return 'Transformer_fPEPS_Model_GlobalMLP'
    
    # ... (tn_contraction, vamp, forward are reused from DConv2d) ...
    tn_contraction = Transformer_fPEPS_Model_DConv2d.tn_contraction
    vamp = Transformer_fPEPS_Model_DConv2d.vamp
    forward = Transformer_fPEPS_Model_DConv2d.forward


class Transformer_fPEPS_Model_Fourier(nn.Module):
    def __init__(
        self,
        tn,
        max_bond,
        nn_eta,
        nn_hidden_dim,
        embed_dim,
        attn_heads,
        attn_depth=1,
        init_perturbation_scale=1e-5,
        dtype=torch.float64,
        **kwargs,
    ):
        super().__init__()
        
        # === 1. TN Setup ===
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        self.ftn_params_pytree = ftn_params_pytree
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)
        
        # === 2. NN Setup ===
        self.nn_hidden_dim = nn_hidden_dim
        self.embed_dim = embed_dim
        
        self.attn_block = SelfAttn_block_pos_batched(
            n_site=len(tn.sites),
            num_classes=tn.phys_dim(),
            embed_dim=self.embed_dim,
            attention_heads=attn_heads,
            depth=attn_depth,
            dtype=self.dtype,
        )
        
        # --- Backflow Generator: Fourier (Spectral) ---
        self.nn_backflow_generator = FourierBackflow(
            lx=self.skeleton.Lx,
            ly=self.skeleton.Ly,
            embed_dim=self.embed_dim,
            hidden_dim=self.nn_hidden_dim,
            param_sizes=self.ftn_params_sizes,
            dtype=self.dtype
        )
        
        self.nn_backflow = nn.Sequential(
            self.attn_block,
            self.nn_backflow_generator
        )

        self.nn_eta = nn_eta
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
        self._init_weights_for_perturbation(scale=init_perturbation_scale)
    
    def _init_weights_for_perturbation(self, scale=1e-5):
        backflow_module = self.nn_backflow_generator
        # FourierBackflow 的最后一层通常是 conv2
        output_layer = backflow_module.conv2
        
        if isinstance(output_layer, (torch.nn.Linear, torch.nn.Conv2d)):
            print(f" -> [Fourier] Clamping output layer weights to scale {scale}")
            torch.nn.init.normal_(output_layer.weight, mean=0.0, std=scale)
            if output_layer.bias is not None:
                torch.nn.init.zeros_(output_layer.bias)
                
        # Init attention
        for m in self.attn_block.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def _get_name(self):
        return 'Transformer_fPEPS_Model_Fourier'

    # ... (Reuse Logic) ...
    tn_contraction = Transformer_fPEPS_Model_DConv2d.tn_contraction
    vamp = Transformer_fPEPS_Model_DConv2d.vamp
    forward = Transformer_fPEPS_Model_DConv2d.forward

# ==============================================================================
# Basic Transformer fPEPS Model without advanced backflow generators
# Old models
# ==============================================================================
import torch.nn.functional as F
class Transformer_fPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, nn_eta, nn_hidden_dim, embed_dim, attn_heads, dtype=torch.float64, **kwargs):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.ftn_params_pytree = ftn_params_pytree

        # register the flat list parameters
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_length = nn.utils.parameters_to_vector(self.ftn_params).shape[0]
        
        self.nn_hidden_dim = nn_hidden_dim
        self.embed_dim = embed_dim
        self.attn_heads = attn_heads
        self.attn = SelfAttn_block_pos(
            n_site=len(tn.sites),
            num_classes=tn.phys_dim(),
            embed_dim=self.embed_dim,
            attention_heads=self.attn_heads,
            dtype=self.dtype,
        )
        # simplest 2 layer MLP
        self.nn_mlp = nn.Sequential(
            nn.Linear(len(tn.sites)*self.embed_dim, self.nn_hidden_dim, dtype=self.dtype),
            nn.GELU(),
            nn.Linear(self.nn_hidden_dim, self.ftn_params_length, dtype=self.dtype),
        )

        # combine attn and mlp into a single nn_backflow
        self.nn_backflow = nn.Sequential(
            self.attn,
            nn.Flatten(start_dim=0),
            self.nn_mlp,
        )
        self.nn_eta = nn_eta

        # We use named_parameters() because self.params only contains parameters, not buffers.
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # combine ftn_params and nn_backflow params into a single pytree
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
    def amplitude(self, x, params):
        # split params into ftn_params and nn_backflow params
        ftn_params = params[:len(self.ftn_params)]
        nn_params = params[len(self.ftn_params):]

        nn_params_dict = dict(zip(self.nn_param_names, nn_params))
        # compute nn_backflow output
        # self.nn_backflow.load_state_dict({k: v for k, v in zip(self.nn_backflow.state_dict().keys(), nn_params)})
        nn_output = torch.func.functional_call(self.nn_backflow, nn_params_dict, x.to(self.dtype))
        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            param = nnftn_params_vector[pointer:pointer+length].view(shape)
            nnftn_params.append(param)
            pointer += length
        nnftn_params = qu.utils.tree_unflatten(nnftn_params, self.ftn_params_pytree)

        tn = qtn.unpack(nnftn_params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()
    
    def vamp(self, x, params):
        return torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)


class NN_fPEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, nn_eta, nn_hidden_dim, dtype=torch.float64):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()
        
        params, skeleton = qtn.pack(tn)
        self.dtype = dtype
        self.skeleton = skeleton
        self.chi = max_bond
        # for torch, further flatten pytree into a single list
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.ftn_params_pytree = ftn_params_pytree

        # register the flat list parameters
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_length = nn.utils.parameters_to_vector(self.ftn_params).shape[0]
        
        self.nn_hidden_dim = nn_hidden_dim
        # simplest 2 layer MLP
        self.nn_backflow = nn.Sequential(
            nn.Linear(len(tn.sites), self.nn_hidden_dim, dtype=self.dtype),
            nn.GELU(),
            nn.Linear(self.nn_hidden_dim, self.ftn_params_length, dtype=self.dtype),
        )
        self.nn_eta = nn_eta

        # We use named_parameters() because self.params only contains parameters, not buffers.
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # combine ftn_params and nn_backflow params into a single pytree
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))
        
    def amplitude(self, x, params):
        # split params into ftn_params and nn_backflow params
        ftn_params = params[:len(self.ftn_params)]
        nn_params = params[len(self.ftn_params):]

        nn_params_dict = dict(zip(self.nn_param_names, nn_params))
        # compute nn_backflow output
        # self.nn_backflow.load_state_dict({k: v for k, v in zip(self.nn_backflow.state_dict().keys(), nn_params)})
        nn_output = torch.func.functional_call(self.nn_backflow, nn_params_dict, x.to(self.dtype))
        ftn_params_vector = nn.utils.parameters_to_vector(ftn_params)
        nnftn_params_vector = ftn_params_vector + self.nn_eta * nn_output
        nnftn_params = []
        pointer = 0
        for shape in self.ftn_params_shape:
            length = torch.prod(torch.tensor(shape)).item()
            param = nnftn_params_vector[pointer:pointer+length].view(shape)
            nnftn_params.append(param)
            pointer += length
        nnftn_params = qu.utils.tree_unflatten(nnftn_params, self.ftn_params_pytree)

        tn = qtn.unpack(nnftn_params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({tn.site_ind(site): x[i] for i, site in enumerate(tn.sites)})
        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[0, amp.Ly//2-1])
            amp.contract_boundary_from_ymax_(max_bond=self.chi, cutoff=0.0, yrange=[amp.Ly//2, amp.Ly-1])
        return amp.contract()
    
    def vamp(self, x, params):
        return torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)


# ==============================================================================
# 1. Original Self-Attention Block (Strict Reproduction)
# ==============================================================================
class SelfAttn_block(nn.Module):
    """ Plain self-attention block with one-hot embedding and layer norm"""
    def __init__(
        self, n_site, num_classes, embedding_dim, attention_heads, dtype=torch.float32
    ):
        super(SelfAttn_block, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Linear layer to project one-hot vectors to the embedding dimension
        self.embedding = nn.Linear(num_classes, embedding_dim)

        # Self-attention block
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=attention_heads, batch_first=True
        )

        self.dtype = dtype
        self.embedding.to(dtype=dtype)
        self.self_attention.to(dtype=dtype)

    def forward(self, input_seq):

        # --- Step 1: Vmap-safe One-hot Encoding ---
        # 1. Prepare output container
        out_shape = input_seq.shape + (self.num_classes,)
        one_hot_encoded = torch.zeros(out_shape, device=input_seq.device, dtype=self.dtype)
        # 2. Prepare indices (avoid .long() inside vmap if possible, or do it safely)
        indices = input_seq.unsqueeze(-1)
        if indices.dtype != torch.int64:
             indices = indices.to(torch.int64)

        # 3. Scatter (Out-of-place for vmap safety)
        one_hot_encoded = one_hot_encoded.scatter(-1, indices, 1.0)

        # Step 2: Embed the one-hot encoded sequence
        embedded = self.embedding(one_hot_encoded) 

        # Step 3: Pass through the self-attention block
        # Note: input_seq is (Batch, Seq_Len), embedded is (Batch, Seq_Len, Embed_Dim)
        attn_output, _ = self.self_attention(
            embedded, embedded, embedded, need_weights=False
        )

        # Step 4: Residual connection and layer normalization
        # Strict reproduction of user's logic: using dynamic shape slicing
        attn_output = F.layer_norm(attn_output + embedded, attn_output.size()[1:])

        return attn_output

# ==============================================================================
# 2. Tensorwise MLP Backflow Module
# ==============================================================================
class TensorwiseMLPBackflow(nn.Module):
    """
    For each on-site tensor, assign a narrow on-site projector MLP.
    Input: Flattened Attention Output (Global Context)
    Output: Tensor Parameters correction
    """
    def __init__(self, input_dim, hidden_dim, param_sizes, dtype):
        super().__init__()
        self.input_dim = input_dim
        self.dtype = dtype
        
        # Use ModuleList instead of ModuleDict for easier batch processing/iteration
        # Each element corresponds to one tensor in the TN
        self.mlps = nn.ModuleList()
        
        for p_size in param_sizes:
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, p_size),
            )
            # Ensure dtype
            mlp.to(dtype=dtype)
            self.mlps.append(mlp)
    
    def initialize_output_scale(self, scale):
        print(f" -> [Init] TensorwiseMLPBackflow: Clamping output weights to scale {scale}")
        for net in self.mlps:
            # The last layer of the MLP is at index 2
            last_layer = net[-1]
            torch.nn.init.normal_(last_layer.weight, mean=0.0, std=scale)
            if last_layer.bias is not None:
                torch.nn.init.zeros_(last_layer.bias)

    def forward(self, attn_features):
        # attn_features: (Batch, N_sites, Embed_Dim)
        B = attn_features.shape[0]
        
        # Flatten the features: (Batch, N_sites * Embed_Dim)
        # This matches the user's logic: "nn_features.view(-1)" applied per sample
        flat_features = attn_features.view(B, -1)
        
        outputs = []
        # Iterate over each unique MLP for each tensor
        for mlp in self.mlps:
            # mlp output: (Batch, p_size)
            outputs.append(mlp(flat_features))
            
        # Concatenate all corrections: (Batch, Total_TN_Params)
        return torch.cat(outputs, dim=1)

# ==============================================================================
# 3. Main Vmap-Compatible Model
# ==============================================================================
class fTN_backflow_attn_Tensorwise_Model_vmap(BasefPEPSBackflowModel):
    def __init__(
        self, 
        ftn, 
        max_bond=None, 
        embed_dim=32, 
        attn_heads=4, 
        nn_hidden_dim=4, 
        nn_eta=1.0, 
        dtype=torch.float32,
        init_perturbation_scale=1e-5,
        **kwargs
    ):
        super().__init__(ftn, max_bond, nn_eta, dtype, kwargs.get('jitter_svd', 0), kwargs.get('debug_file'))
        self.dtype = dtype
        
        # --- 1. TN Parameter Setup ---
        # Extract raw arrays and skeleton
        params, skeleton = qtn.pack(ftn)
        self.skeleton = skeleton
        self.chi = max_bond
        
        # Flatten the nested dictionary structure into a single list of tensors
        # This is crucial for vmap to work (it can't trace ModuleDict)
        ftn_params_flat, ftn_params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.ftn_params_pytree = ftn_params_pytree

        # Register TN parameters
        self.ftn_params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in ftn_params_flat
        ])
        
        # Store shapes and sizes to reconstruct/slice later
        self.ftn_params_shape = [p.shape for p in self.ftn_params]
        self.ftn_params_sizes = [p.numel() for p in self.ftn_params] 
        self.ftn_params_length = sum(self.ftn_params_sizes)

        # --- 2. Neural Network Setup ---
        input_dim = ftn.Lx * ftn.Ly
        phys_dim = ftn.phys_dim()
        self.embedding_dim = embed_dim
        
        # A. Attention Block
        self.attn_block = SelfAttn_block(
            n_site=input_dim,
            num_classes=phys_dim,
            embedding_dim=embed_dim,
            attention_heads=attn_heads,
            dtype=self.dtype
        )
        
        # B. Tensorwise MLPs
        # Input to MLP is Flattened Attn Output: Lx * Ly * Embed_Dim
        mlp_input_dim = input_dim * embed_dim
        
        self.nn_backflow_generator = TensorwiseMLPBackflow(
            input_dim=mlp_input_dim,
            hidden_dim=nn_hidden_dim, # Matches user's nn_hidden_dim
            param_sizes=self.ftn_params_sizes,
            dtype=self.dtype
        )
        
        # Combine into sequential
        self.nn_backflow = nn.Sequential(
            self.attn_block,
            self.nn_backflow_generator
        )

        self.nn_eta = nn_eta
        
        # Helper for functional call
        self.nn_param_names = [name for name, _ in self.nn_backflow.named_parameters()]
        
        # Combine all parameters for optimizers
        self.params = nn.ParameterList(list(self.ftn_params) + list(self.nn_backflow.parameters()))

        # Initialize the MLP output layers to small random values
        self.nn_backflow_generator.initialize_output_scale(scale=init_perturbation_scale)