import quimb.tensor as qtn
import torch
import torch.nn as nn
import cotengra as ctg
import time
import quimb as qu
from quimb.tensor.tensor_core import tags_to_oset, group_inds

from vmc_torch.experiment.tn_model import wavefunctionModel


# VMAP-compatible PEPS model
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

class circuit_TNF_2d_Model(nn.Module):
    """
    ham: qtn.LocalHam2D
    """
    def __init__(
        self,
        tns,
        ham,
        trotter_tau,
        depth,
        # second_order_reflect=False,
        mode='projector3d',
        from_which="zmax",
        max_bond=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.trotter_tau = trotter_tau
        self.depth = depth
        # self.second_order_reflect = second_order_reflect
        self.max_bond = max_bond if (type(max_bond) is int and max_bond > 0) else None
        if self.max_bond is None:
            self.tree = None
        self.from_which = from_which
        self.mode = mode
        self.site_inds = tns.site_inds

        self.ham = ham
        circuit_tnf = self.form_gated_tns_tnf(
            tns,
            ham,
            depth,
            tau=trotter_tau,
            contract_layer=False if mode=='SVDU' else True,
            gate_contract=False if mode=='SVDU' else 'split-gate',
        )
        if mode != 'SVDU':
            self.Lx, self.Ly, self.Lz = circuit_tnf.Lx, circuit_tnf.Ly, circuit_tnf.Lz

        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(circuit_tnf)

        # for torch, further flatten pytree into a single list
        params_flat, params_pytree = qu.utils.tree_flatten(
            params, get_ref=True
        )
        self.params_pytree = params_pytree

        # register the flat list parameters
        self.params = torch.nn.ParameterList([
            torch.as_tensor(x, dtype=self.dtype) for x in params_flat
        ])

    def form_gated_tns_tnf(
        self,
        tns,
        ham,
        depth,
        tau=0.5,
        nn_where_list=None,
        d_tag_id="ROUND_{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        z_tag_id="ROUND_{}",
        site_tag_id_2d="I{},{}",
        site_tag_id_3d="I{},{},{}",
        gate_contract='split-gate',
        contract_layer=True
    ):
        import itertools

        tns1 = tns.copy()

        if not isinstance(nn_where_list, list) and not isinstance(nn_where_list, tuple):
            Warning(
                "nn_where_list is not a list of tuples, using auto ordering from the Hamiltonian"
            )
            nn_where_list = ham.get_auto_ordering()

        # Change tags for the initial tns
        for ts in tns1.tensors:
            ts.modify(tags=["ROUND_0"] + list(ts.tags))

        # Apply the gates and add corresponding tags
        for i in range(depth):
            for where in nn_where_list:
                gate = ham.get_gate_expm(where, -1 * tau)
                site_inds = [tns1.site_ind_id.format(*site) for site in where]
                extra_tags = ["ROUND_{}".format(i + 1), where]
                ltag = tns1.site_tag_id.format(*where[0])
                rtag = tns1.site_tag_id.format(*where[1])
                tns1 = tns1.gate_inds(
                    gate,
                    inds=site_inds,
                    contract=gate_contract if i < depth - 1 else 'split-gate',
                    tags=extra_tags,
                    ltags=ltag,
                    rtags=rtag,
                )

        if contract_layer:
            # Contract the gates in each round to a TNO
            for i in range(1, depth + 1):
                for site in tns1.sites:
                    tns1.contract_tags_(
                        [tns1.site_tag_id.format(*site), f"ROUND_{i}"],
                        inplace=True,
                        which="all",
                    )

            # Add site tags
            for d in range(0, depth + 1):
                for x, y in itertools.product(range(tns1.Lx), range(tns1.Ly)):
                    # print(x,y)
                    ts = tns1[[site_tag_id_2d.format(x, y), d_tag_id.format(d)]]
                    ts.add_tag(x_tag_id.format(x))
                    ts.add_tag(y_tag_id.format(y))
                    ts.add_tag(z_tag_id.format(d))
                    ts.add_tag(site_tag_id_3d.format(x, y, d))

            circuit_tnf = PEPS_TNF.from_TN(tns1)
            circuit_tnf.set_depth(depth)

            return circuit_tnf
        else:
            return tns1

    
    def amplitude(self, x, params):
        tn = qtn.unpack(params, self.skeleton)
        # might need to specify the right site ordering here
        amp = tn.isel({ind: x[i] for i, ind in enumerate(self.site_inds)})
        if self.mode != 'SVDU':
            amp.view_as_(
                qtn.PEPS3D,
                site_ind_id="k{},{}",
                x_tag_id="X{}",
                y_tag_id="Y{}", 
                z_tag_id="ROUND_{}",
                site_tag_id="I{},{},{}",
                Lx=self.Lx,
                Ly=self.Ly,
                Lz=self.Lz,
            )
        if self.max_bond is None:
            # if self.tree is None:
            #     opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
            #     self.tree = amp.contraction_tree(optimize=opt)
            # amp_val = amp.contract(optimize=self.tree)
            amp_val = amp.contract()
        else:
            if self.mode == 'peps':
                amp = amp.contract_boundary_from(
                    from_which=self.from_which,
                    max_bond=self.max_bond,
                    cutoff=0.0,
                    zrange=(0, self.Lz - 1),
                    yrange=[0, self.Ly - 1],
                    xrange=(0, self.Lx - 1),
                    mode="peps",
                )
                amp_val = amp.contract()
            elif self.mode == 'projector3d':
                amp = amp.contract_boundary(
                    max_bond=self.max_bond,
                    cutoff=0.0,
                    canonize=False,
                    mode="projector3d",
                    equalize_norms=1.0,
                    sequence=[self.from_which] * amp.Lz + ["ymax"] * amp.Ly,
                    max_separation=0,
                    final_contract_opts=dict(strip_exponent=True),
                    progbar=False,
                )
                mantissa, exponent = amp
                amp_val = mantissa * 10**exponent
            elif self.mode == 'hotrg':
                amp_val = amp.contract_hotrg(
                    max_bond=self.max_bond,
                    cutoff=0.0,
                    canonize=False,
                    canonize_opts=None,
                    sequence=("x", "y", "z"),
                    max_separation=1,
                    max_unfinished=1,
                    lazy=False,
                    equalize_norms=False,
                    final_contract=True,
                    final_contract_opts=None,
                    progbar=False,
                    inplace=False,
                )
            elif self.mode == 'SVDU':
                depth = self.depth
                for site in amp.sites:
                    amp.contract_tags_(
                        [amp.site_tag_id.format(*site), f"ROUND_{depth}"],
                        inplace=True,
                        which="all",
                    )
                main_tnf, inverse_peps = amp.partition(f'ROUND_{depth}', inplace=True)
                gates_tn, init_peps = main_tnf.partition(f'ROUND_{0}', inplace=True)
                init_outer_inds = qtn.bonds(gates_tn, init_peps)
                init_peps.reindex_(dict(zip(init_outer_inds, self.site_inds)))

                inverse_peps.outer_inds(), inverse_peps.tensors
                outer_inds = inverse_peps.outer_inds()
                inverse_peps.reindex_({outer_inds[i]: self.site_inds[i] for i in range(len(outer_inds))})
                main_tnf.reindex_({outer_inds[i]: self.site_inds[i] for i in range(len(outer_inds))})
                inverse_peps.view_as_(qtn.PEPS)
                nn_where_list = self.ham.get_auto_ordering()
                for d in range(depth-1, 0, -1):
                    for where in nn_where_list[::-1]:
                        (tid,) = gates_tn._get_tids_from_tags([f'ROUND_{d}', where])
                        gate = gates_tn.pop_tensor(tid)
                        pinds = [inverse_peps.site_ind_id.format(*site) for site in where]
                        gate_inds_outer = [ind for ind in gate.inds if ind not in pinds]
                        (tid1,), (tid2,) = inverse_peps._get_tids_from_inds(pinds[0]), inverse_peps._get_tids_from_inds(pinds[1])

                        temp_tags = tags_to_oset(['gate-temp'])
                        inverse_peps.tensor_map[tid1].modify(tags = inverse_peps.tensor_map[tid1].tags | temp_tags)
                        inverse_peps.tensor_map[tid2].modify(tags = inverse_peps.tensor_map[tid2].tags | temp_tags)
                        tl, tr = inverse_peps._inds_get(pinds[0], pinds[1])
                        bnds_l, (bix,), bnds_r = group_inds(tl, tr)

                        gate.modify(tags = gate.tags | temp_tags)

                        inverse_peps.gate_inds_with_tn_(inds=pinds, gate=gate, gate_inds_inner=pinds, gate_inds_outer=gate_inds_outer)
                        gates_tn.reindex_(dict(zip(gate_inds_outer, pinds)))
                        # inverse_peps.draw('gate-temp')
                        inverse_peps.contract_tags_(temp_tags)
                        (contracted_ts_tid,) = inverse_peps._get_tids_from_tags(temp_tags)
                        

                        contracted_ts = inverse_peps.pop_tensor(contracted_ts_tid)
                        tln, *maybe_svals, trn = contracted_ts.split(
                            left_inds=bnds_l,
                            right_inds=bnds_r,
                            bond_ind=bix,
                            get="tensors",
                            max_bond=self.max_bond,
                            cutoff=0.0,
                            absorb='left'
                        )
                        # remove temp tags
                        tln.tags.remove('gate-temp')
                        trn.tags.remove('gate-temp')
                        inverse_peps = (inverse_peps | tln | trn)
                        inverse_peps.view_like_(amp)
                amp_val = (init_peps | inverse_peps).contract()
        return amp_val
    
    def vamp(self, x, params):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)
    

# === old non-VMAP-compatible TNF models === #
class PEPS_TNF(qtn.PEPS):
    def __init__(self, arrays, depth=None, Lx=None, Ly=None, *args, **kwargs):
        if isinstance(arrays, PEPS_TNF):
            self.Lz = arrays.Lz
            super().__init__(arrays, *args, **kwargs)
            return
        self.Lz = depth + 1
        self.depth = depth
        super().__init__(arrays, *args, **kwargs)

    def set_depth(self, depth):
        self.Lz = depth + 1
        self.depth = depth

    def get_amp(self, config, inplace=False, conj=True, reverse=1):
        """Get the amplitude of a configuration in a PEPS."""
        tnf = self if inplace else self.copy()
        amp = tnf.isel({tnf.site_inds[i]: int(s) for i, s in enumerate(config)})
        amp.view_as_(
            qtn.PEPS3D,
            site_ind_id="k{},{}",
            x_tag_id="X{}",
            y_tag_id="Y{}", 
            z_tag_id="ROUND_{}",
            site_tag_id="I{},{},{}",
            Lx=tnf.Lx,
            Ly=tnf.Ly,
            Lz=tnf.Lz,
        )
        return amp


class MPS_TNF(qtn.MatrixProductState):
    def __init__(self, arrays, depth=None, L=None, *args, **kwargs):
        if isinstance(arrays, MPS_TNF):
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
        # product_state = self.product_bra_state(config, reverse=reverse).conj() if conj else self.product_bra_state(config, reverse=reverse)
        
        # amp = tnf|product_state # ---T---<---|n>

        amp = tnf.isel({tnf.site_inds[i]: int(s) for i, s in enumerate(config)})
        
        # for ind in tnf.site_inds:
        #     amp.contract_ind(ind)

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
    
class circuit_TNF(wavefunctionModel):
    def __init__(
        self,
        mps,
        ham,
        trotter_tau,
        depth,
        second_order_reflect=False,
        from_which="ymin",
        max_bond=None,
        dtype=torch.float32,
    ):
        super().__init__(dtype)
        self.param_dtype = dtype
        self.trotter_tau = trotter_tau
        self.depth = depth
        self.second_order_reflect = second_order_reflect
        self.max_bond = max_bond if (type(max_bond) is int and max_bond > 0) else None
        self.from_which = from_which

        self.ham = ham
        circuit_tnf = self.form_gated_mps_tnf(
            mps,
            ham,
            depth,
            tau=trotter_tau,
        )

        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(circuit_tnf)

        self.torch_tn_params = {
            str(tid): nn.Parameter(data)  # Convert Tensor to Parameter
            for tid, data in params.items()
        }
        # register the torch tensors as parameters
        for tid, param in self.torch_tn_params.items():
            self.register_parameter(tid, param)

    def form_gated_mps_tnf(
        self,
        mps,
        ham,
        depth,
        tau=0.5,
        nn_where_list=None,
        x_tag_id="ROUND_{}",
        y_tag_id="I{}",
        site_tag_id="I{},{}",
    ):
        mps1 = mps.copy()

        if not isinstance(nn_where_list, list):
            Warning(
                "nn_where_list is not a list of tuples, using all nearest neighbor terms in the Hamiltonian"
            )
            nn_where_list = [(i, i + 1) for i in range(mps.L - 1)]

        # Change tags for the initial MPS
        for ts in mps1.tensors:
            ts.modify(tags=["ROUND_0"] + list(ts.tags))

        # Apply the gates and add corresponding tags
        for i in range(depth):
            for where in nn_where_list:
                gate = ham.get_gate_expm(where, -1 * tau)
                site_inds = [mps1.site_ind_id.format(site) for site in where]
                extra_tags = ["ROUND_{}".format(i + 1)]
                ltag = mps1.site_tag_id.format(where[0])
                rtag = mps1.site_tag_id.format(where[1])
                mps1 = mps1.gate_inds(
                    gate,
                    inds=site_inds,
                    contract="split-gate",
                    tags=extra_tags,
                    ltags=ltag,
                    rtags=rtag,
                )

        # Contract the gates in each round to a MPO
        for i in range(1, depth + 1):
            for site in mps1.sites:
                mps1.contract_tags_(
                    [mps1.site_tag_id.format(site), f"ROUND_{i}"],
                    inplace=True,
                    which="all",
                )

        # Add site tags
        for x in range(0, depth + 1):
            for y in range(mps1.L):
                ts = mps1[[x_tag_id.format(x), y_tag_id.format(y)]]
                ts.add_tag(site_tag_id.format(x, y))

        circuit_tnf = MPS_TNF.from_TN(mps1)
        circuit_tnf.set_Lx(depth + 1)
        circuit_tnf.set_Ly(mps1.L)

        return circuit_tnf

    def get_state(self):
        params = {int(tid): data for tid, data in self.torch_tn_params.items()}
        peps = qtn.unpack(params, self.skeleton)
        return peps

    def amplitude(self, x):
        psi = self.get_state()
        batch_amps = []
        for x_i in x:
            if not isinstance(x_i, torch.Tensor):
                x_i = torch.tensor(x_i, dtype=self.param_dtype)

            amp = psi.get_amp(x_i)

            if self.max_bond is None:
                amp_val = amp.contract()
            else:
                if self.from_which == "ymin":
                    amp = amp.contract_boundary_from_ymin(
                        max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly - 1]
                    )
                elif self.from_which == "ymax":
                    amp = amp.contract_boundary_from_ymax(
                        max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly - 1]
                    )
                elif self.from_which == "yboth":
                    # print('Using MPS contraction scheme from yboth')
                    amp = amp.contract_boundary_from_ymin(
                        max_bond=self.max_bond, cutoff=0.0, yrange=[0, psi.Ly // 2 - 1]
                    )
                    amp = amp.contract_boundary_from_ymax(
                        max_bond=self.max_bond,
                        cutoff=0.0,
                        yrange=[psi.Ly // 2, psi.Ly - 1],
                    )
                elif self.from_which == "xmax":
                    amp = amp.contract_boundary_from_xmax(
                        max_bond=self.max_bond, cutoff=0.0, xrange=[0, psi.Lx - 1]
                    )
                elif self.from_which == "xmin":
                    print("Using MPS contraction scheme from xmin")
                    amp = amp.contract_boundary_from_xmin(
                        max_bond=self.max_bond, cutoff=0.0, xrange=[0, psi.Lx - 1]
                    )
                else:
                    raise ValueError(
                        "from_which should be one of ymin, ymax, yboth, xmax, xmin"
                    )

                amp_val = amp.contract()
            if amp_val == 0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)

        return torch.stack(batch_amps)


class circuit_TNF_2d(wavefunctionModel):
    def __init__(
        self,
        tns,
        ham,
        trotter_tau,
        depth,
        # second_order_reflect=False,
        mode='projector3d',
        from_which="zmax",
        max_bond=None,
        dtype=torch.float32,
    ):
        super().__init__(dtype)
        self.param_dtype = dtype
        self.trotter_tau = trotter_tau
        self.depth = depth
        # self.second_order_reflect = second_order_reflect
        self.max_bond = max_bond if (type(max_bond) is int and max_bond > 0) else None
        if self.max_bond is None:
            self.tree = None
        self.from_which = from_which
        self.mode = mode

        self.ham = ham
        circuit_tnf = self.form_gated_tns_tnf(
            tns,
            ham,
            depth,
            tau=trotter_tau,
        )

        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(circuit_tnf)

        self.torch_tn_params = {
            str(tid): nn.Parameter(data)  # Convert Tensor to Parameter
            for tid, data in params.items()
        }
        # register the torch tensors as parameters
        for tid, param in self.torch_tn_params.items():
            self.register_parameter(tid, param)

    def get_state(self):
        params = {int(tid): data for tid, data in self.torch_tn_params.items()}
        peps = qtn.unpack(params, self.skeleton)
        return peps

    def form_gated_tns_tnf(
        self,
        tns,
        ham,
        depth,
        tau=0.5,
        nn_where_list=None,
        d_tag_id="ROUND_{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        z_tag_id="ROUND_{}",
        site_tag_id_2d="I{},{}",
        site_tag_id_3d="I{},{},{}",
    ):
        tns1 = tns.copy()

        if not isinstance(nn_where_list, list) and not isinstance(nn_where_list, tuple):
            Warning(
                "nn_where_list is not a list of tuples, using auto ordering from the Hamiltonian"
            )
            nn_where_list = ham.get_auto_ordering()

        # Change tags for the initial tns
        for ts in tns1.tensors:
            ts.modify(tags=["ROUND_0"] + list(ts.tags))

        # Apply the gates and add corresponding tags
        for i in range(depth):
            for where in nn_where_list:
                gate = ham.get_gate_expm(where, -1 * tau)
                site_inds = [tns1.site_ind_id.format(*site) for site in where]
                extra_tags = ["ROUND_{}".format(i + 1)]
                ltag = tns1.site_tag_id.format(*where[0])
                rtag = tns1.site_tag_id.format(*where[1])
                tns1 = tns1.gate_inds(
                    gate,
                    inds=site_inds,
                    contract="split-gate",
                    tags=extra_tags,
                    ltags=ltag,
                    rtags=rtag,
                )

        # Contract the gates in each round to a MPO
        for i in range(1, depth + 1):
            for site in tns1.sites:
                tns1.contract_tags_(
                    [tns1.site_tag_id.format(*site), f"ROUND_{i}"],
                    inplace=True,
                    which="all",
                )
        import itertools

        # Add site tags
        for d in range(0, depth + 1):
            for x, y in itertools.product(range(tns1.Lx), range(tns1.Ly)):
                # print(x,y)
                ts = tns1[[site_tag_id_2d.format(x, y), d_tag_id.format(d)]]
                ts.add_tag(x_tag_id.format(x))
                ts.add_tag(y_tag_id.format(y))
                ts.add_tag(z_tag_id.format(d))
                ts.add_tag(site_tag_id_3d.format(x, y, d))

        # return tns1

        circuit_tnf = PEPS_TNF.from_TN(tns1)
        circuit_tnf.set_depth(depth)

        return circuit_tnf

    def amplitude(self, x):
        psi = self.get_state()
        batch_amps = []
        for x_i in x:
            if not isinstance(x_i, torch.Tensor):
                x_i = torch.tensor(x_i, dtype=self.param_dtype)

            amp = psi.get_amp(x_i)
            try:
                if self.max_bond is None:
                    if self.tree is None:
                        opt = ctg.HyperOptimizer(progbar=True, max_repeats=10, parallel=True)
                        self.tree = amp.contraction_tree(optimize=opt)
                    amp_val = amp.contract(optimize=self.tree)
                else:
                    if self.from_which == "zmin":
                        if self.mode == 'peps':
                            amp = amp.contract_boundary_from(
                                from_which="zmin",
                                max_bond=self.max_bond,
                                cutoff=0.0,
                                zrange=(0, psi.Lz - 1),
                                yrange=[0, psi.Ly - 1],
                                xrange=(0, psi.Lx - 1),
                                mode="peps",
                            )
                            amp_val = amp.contract()
                        elif self.mode == 'projector3d':
                            amp = amp.contract_boundary(
                                max_bond=self.max_bond,
                                cutoff=0.0,
                                canonize=False,
                                mode="projector3d",
                                equalize_norms=1.0,
                                sequence=["zmin"] * amp.Lz + ["ymax"] * amp.Ly,
                                max_separation=0,
                                final_contract_opts=dict(strip_exponent=True),
                                progbar=False,
                            )
                            mantissa, exponent = amp
                            amp_val = mantissa * 10**exponent
                            

                    elif self.from_which == "zmax":
                        if self.mode == 'peps':
                            amp = amp.contract_boundary_from(
                                from_which="zmax",
                                max_bond=self.max_bond,
                                cutoff=0.0,
                                zrange=(0, psi.Lz - 1),
                                yrange=[0, psi.Ly - 1],
                                xrange=(0, psi.Lx - 1),
                                mode="peps",
                            )
                            amp_val = amp.contract()
                        elif self.mode == 'projector3d':
                            amp = amp.contract_boundary(
                                max_bond=self.max_bond,
                                cutoff=0.0,
                                canonize=False,
                                mode="projector3d",
                                equalize_norms=1.0,
                                sequence=["zmax"] * amp.Lz + ["ymax"] * amp.Ly,
                                max_separation=0,
                                final_contract_opts=dict(strip_exponent=True),
                                progbar=False,
                            )
                            mantissa, exponent = amp
                            amp_val = mantissa * 10**exponent
                        elif self.mode == 'hotrg':
                            amp_val = amp.contract_hotrg(
                                max_bond=self.max_bond,
                                cutoff=0.0,
                                canonize=False,
                                canonize_opts=None,
                                sequence=("x", "y", "z"),
                                max_separation=1,
                                max_unfinished=1,
                                lazy=False,
                                equalize_norms=False,
                                final_contract=True,
                                final_contract_opts=None,
                                progbar=False,
                                inplace=False,
                            )
                    else:
                        if self.mode == 'peps':
                            amp = amp.contract_boundary_from(
                                from_which=self.from_which,
                                max_bond=self.max_bond,
                                cutoff=0.0,
                                zrange=(0, psi.Lz - 1),
                                yrange=[0, psi.Ly - 1],
                                xrange=(0, psi.Lx - 1),
                                mode="peps",
                            )
                            amp_val = amp.contract()
                        elif self.mode == 'projector3d':
                            axis_set = ['x','y','z']
                            axis_map = {'x':psi.Lx, 'y':psi.Ly, 'z':psi.Lz}
                            axis = self.from_which[0]
                            axis_set.remove(axis)
                            final_contraction_axis = axis_set[0]

                            sequence0 = [self.from_which]*axis_map[axis]
                            sequence1 = [final_contraction_axis+"max"]*axis_map[final_contraction_axis]

                            sequence = sequence0 + sequence1
                            
                            amp = amp.contract_boundary(
                                max_bond=self.max_bond,
                                cutoff=0.0,
                                canonize=False,
                                mode="projector3d",
                                equalize_norms=1.0,
                                sequence=sequence,
                                max_separation=0,
                                final_contract_opts=dict(strip_exponent=True),
                                progbar=False,
                            )
                            mantissa, exponent = amp
                            amp_val = mantissa * 10**exponent
            except Exception as e:
                print(e)
                amp_val = amp.contract()
                print("Contraction failed, using exact contraction instead")
                
                
            if amp_val == 0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)
        return torch.stack(batch_amps)



class circuit_TNF_2d_SU(wavefunctionModel):
    def __init__(
        self,
        tns,
        ham,
        trotter_tau,
        depth,
        second_order_reflect=False,
        max_bond=None,
        profile_time=False,
        dtype=torch.float32,
    ):
        super().__init__(dtype)
        self.param_dtype = dtype
        self.trotter_tau = trotter_tau
        self.second_order_reflect = second_order_reflect
        self.depth = depth
        self.profile_time = profile_time
        # self.second_order_reflect = second_order_reflect
        self.max_bond = max_bond if (type(max_bond) is int and max_bond > 0) else None
        if self.max_bond is None:
            self.tree = None

        self.ham = ham

        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(tns)

        self.torch_tn_params = {
            str(tid): nn.Parameter(data)  # Convert Tensor to Parameter
            for tid, data in params.items()
        }
        # register the torch tensors as parameters
        for tid, param in self.torch_tn_params.items():
            self.register_parameter(tid, param)

    def get_state(self):
        params = {int(tid): data for tid, data in self.torch_tn_params.items()}
        peps = qtn.unpack(params, self.skeleton)
        return peps

    def form_gated_tns_tnf(
        self,
        tns,
        ham,
        depth,
        tau=0.5,
        nn_where_list=None,
        d_tag_id="ROUND_{}",
        x_tag_id="X{}",
        y_tag_id="Y{}",
        z_tag_id="ROUND_{}",
        site_tag_id_2d="I{},{}",
        site_tag_id_3d="I{},{},{}",
    ):
        tns1 = tns.copy()

        if not isinstance(nn_where_list, list) and not isinstance(nn_where_list, tuple):
            Warning(
                "nn_where_list is not a list of tuples, using auto ordering from the Hamiltonian"
            )
            nn_where_list = ham.get_auto_ordering()

        # Change tags for the initial tns
        for ts in tns1.tensors:
            ts.modify(tags=["ROUND_0"] + list(ts.tags))

        # Apply the gates and add corresponding tags
        for i in range(depth):
            for where in nn_where_list:
                gate = ham.get_gate_expm(where, -1 * tau)
                site_inds = [tns1.site_ind_id.format(*site) for site in where]
                extra_tags = ["ROUND_{}".format(i + 1)]
                ltag = tns1.site_tag_id.format(*where[0])
                rtag = tns1.site_tag_id.format(*where[1])
                tns1 = tns1.gate_inds(
                    gate,
                    inds=site_inds,
                    contract="split-gate",
                    tags=extra_tags,
                    ltags=ltag,
                    rtags=rtag,
                )

        # Contract the gates in each round to a MPO
        for i in range(1, depth + 1):
            for site in tns1.sites:
                tns1.contract_tags_(
                    [tns1.site_tag_id.format(*site), f"ROUND_{i}"],
                    inplace=True,
                    which="all",
                )
        import itertools

        # Add site tags
        for d in range(0, depth + 1):
            for x, y in itertools.product(range(tns1.Lx), range(tns1.Ly)):
                # print(x,y)
                ts = tns1[[site_tag_id_2d.format(x, y), d_tag_id.format(d)]]
                ts.add_tag(x_tag_id.format(x))
                ts.add_tag(y_tag_id.format(y))
                ts.add_tag(z_tag_id.format(d))
                ts.add_tag(site_tag_id_3d.format(x, y, d))

        # return tns1

        circuit_tnf = PEPS_TNF.from_TN(tns1)
        circuit_tnf.set_depth(depth)

        return circuit_tnf


    def amplitude(self, x):
        psi = self.get_state()
        batch_amps = []
        for x_i in x:
            if not isinstance(x_i, torch.Tensor):
                x_i = torch.tensor(x_i, dtype=self.param_dtype)
            site_maps = dict((psi.sites[i], torch.tensor([1,0] if config == 0 else torch.tensor([0,1]))) for i, config in enumerate(x_i))
            config_product_state = qtn.PEPS.product_state(site_maps)
            config_product_state.apply_to_arrays(lambda x: x.type(self.param_dtype))
            # try:
            # Manual update 
            gauges = {}
            config_product_state.gauge_all_simple_(max_iterations=1, gauges=gauges)
            for _ in range(self.depth):
                t0 = time.time()
                for where in self.ham.get_auto_ordering()[::-1]:
                    G = self.ham.get_gate_expm(where, -1 * self.trotter_tau)
                    config_product_state.gate_simple_(
                        G,
                        where,
                        max_bond=self.max_bond,
                        gauges=gauges,
                        cutoff=0.0,
                        renorm=False,
                    )
                t1 = time.time()
                config_product_state.gauge_all_simple_(max_iterations=1, gauges=gauges)
                t2 = time.time()
                if self.profile_time:
                    print(f"One layer gate time: {t1-t0}, gauge time: {t2-t1}")

            config_product_state.gauge_simple_insert(gauges)
            two_layer_tn = (config_product_state | psi)
            t0 = time.time()
            two_layer_tn.contract_boundary_from_xmax_(xrange=[0, psi.Lx-1], yrange=[0, psi.Ly-1], max_bond=self.max_bond, canonize=False, mode="projector",equalize_norms=1.0,)
            amp_val = two_layer_tn.contract()
            t1 = time.time()
            if self.profile_time:
                print(f"2-layer TN contraction time: {t1-t0}")

                        
                        # config_product_state.gate_simple_(
                        #     G,
                        #     where,
                        #     max_bond=self.max_bond,
                        #     gauges=gauges,
                        #     cutoff=0.0,
                        #     renorm=False,
                        # )
            # except Exception as e:
            #     print(e)
            #     circuit_tnf = self.form_gated_tns_tnf(
            #         psi,
            #         self.ham,
            #         self.depth,
            #         tau=self.trotter_tau,
            #         nn_where_list=self.ham.get_auto_ordering(),
            #     )
            #     amp = circuit_tnf.get_amp(x_i)
            #     amp = amp.contract_boundary(
            #         max_bond=self.max_bond,
            #         cutoff=0.0,
            #         canonize=False,
            #         mode="projector3d",
            #         equalize_norms=1.0,
            #         sequence=["zmax"] * amp.Lz + ["ymax"] * amp.Ly,
            #         max_separation=0,
            #         final_contract_opts=dict(strip_exponent=True),
            #         progbar=False,
            #     )
            #     mantissa, exponent = amp
            #     amp_val = mantissa * 10**exponent
            #     print("Contraction failed, using projector3d contraction instead")


            if amp_val == 0.0:
                amp_val = torch.tensor(0.0)
            batch_amps.append(amp_val)
        return torch.stack(batch_amps)