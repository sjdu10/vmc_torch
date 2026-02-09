from . import torch, qu, qtn, nn
from . import use_jitter_svd, pack_ftn, unpack_ftn, get_params_ftn

class PEPS_Model(nn.Module):
    def __init__(self, tn, max_bond, dtype=torch.float64):
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


class fPEPS_Model_reuse(nn.Module):

    def __init__(self, tn, max_bond, dtype=torch.float64, contract_boundary_opts={}, **kwargs):
        import quimb as qu
        import quimb.tensor as qtn
        super().__init__()

        params, skeleton = qtn.pack(tn)
        self.contract_boundary_opts = contract_boundary_opts
        self.dtype = dtype
        self.skeleton = skeleton
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.bMPS_x_skeletons = {}
        self.bMPS_y_skeletons = {}
        self.bMPS_params_x_in_dims = None
        self.bMPS_params_y_in_dims = None
        self.chi = max_bond
        self.debug = kwargs.get('debug', False)

        # for torch, further flatten pytree into a single list
        params_flat, params_pytree = qu.utils.tree_flatten(params,
                                                           get_ref=True)
        self.params_pytree = params_pytree

        # register the flat list parameters
        self.params = torch.nn.ParameterList(
            [torch.as_tensor(x, dtype=self.dtype) for x in params_flat])
        
        self.radius = 0

    @torch.no_grad()
    def cache_bMPS_skeleton(self, x):
        params = qu.utils.tree_unflatten(self.params, self.params_pytree)
        tns = qtn.unpack(
            params, self.skeleton
        )  # when unpacking tns, use quimb native unpack function is enough
        amp = tns.isel({
            tns.site_ind(site): x[i]
            for i, site in enumerate(tns.sites)
        })
        env_x = amp.compute_x_environments(max_bond=self.chi, cutoff=0.0, **self.contract_boundary_opts)
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
        del env_x
        del env_y
        del bMPS_params_dict


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
                        xrange=[row_id - 1, row_id],
                        **self.contract_boundary_opts)
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
                        xrange=[row_id, row_id + 1],
                        **self.contract_boundary_opts)
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
                        yrange=[col_id - 1, col_id],
                        **self.contract_boundary_opts)
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
                        yrange=[col_id, col_id + 1],
                        **self.contract_boundary_opts)
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
            if self.chi > 0:
                if len(amp_reuse.tensors) > 2*self.Ly:
                    amp_reuse.contract_boundary_from_xmin_(max_bond=self.chi, cutoff=0.0, xrange=[bMPS_keys[0][1], min(bMPS_keys[0][1]+1, self.Lx-1)], **self.contract_boundary_opts)
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
            if self.chi > 0:
                if len(amp_reuse.tensors) > 2*self.Lx:
                    amp_reuse.contract_boundary_from_ymin_(max_bond=self.chi, cutoff=0.0, yrange=[bMPS_keys[0][1], min(bMPS_keys[0][1]+1, self.Ly-1)], **self.contract_boundary_opts)
            return amp_reuse.contract()

        if self.chi > 0:
            amp.contract_boundary_from_ymin_(max_bond=self.chi,
                                             cutoff=0.0,
                                             yrange=[0, amp.Ly // 2 - 1],
                                             **self.contract_boundary_opts)
            amp.contract_boundary_from_ymax_(max_bond=self.chi,
                                             cutoff=0.0,
                                             yrange=[amp.Ly // 2, amp.Ly - 1],
                                             **self.contract_boundary_opts)

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
            amps = torch.vmap(
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
            if self.debug:
                amp_tn_s = [self.amp_tn(x[i]) for i in range(x.shape[0])]
                amps_benchmark = torch.stack([amp_tn.contract() for amp_tn in amp_tn_s], dim=0)
                assert torch.allclose(
                    torch.tensor(amps),
                    torch.tensor(amps_benchmark),
                    rtol=1e-3,
                    atol=1e-3,
                ), f"Amps with bMPS reuse do not match direct contraction! {amps} vs {amps_benchmark}"
            return amps

        if bMPS_params_ymin is not None and bMPS_params_ymax is not None:
            amps = torch.vmap(
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
            if self.debug:
                amp_tn_s = [self.amp_tn(x[i]) for i in range(x.shape[0])]
                amps_benchmark = torch.stack([amp_tn.contract() for amp_tn in amp_tn_s], dim=0)
                assert torch.allclose(
                    torch.tensor(amps),
                    torch.tensor(amps_benchmark),
                    rtol=1e-3,
                    atol=1e-3,
                ), f"Amps with bMPS reuse do not match direct contraction! {amps} vs {amps_benchmark}"
            return amps

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

