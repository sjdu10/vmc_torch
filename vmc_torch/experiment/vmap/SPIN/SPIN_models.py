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
    def __init__(self, tn, max_bond, dtype=torch.float64, **kwargs):
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
        max_bond_final=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.trotter_tau = trotter_tau
        self.depth = depth
        # self.second_order_reflect = second_order_reflect
        self.max_bond = max_bond if (type(max_bond) is int and max_bond > 0) else None
        self.max_bond_final = max_bond_final if (type(max_bond_final) is int and max_bond_final > 0) else None
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

        self._vamp = torch.vmap(
            self.amplitude,
            in_dims=(0, None),
        )

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

                # inverse_peps.outer_inds(), inverse_peps.tensors
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
                            absorb='left',
                        )
                        # remove temp tags
                        tln.tags.remove('gate-temp')
                        trn.tags.remove('gate-temp')
                        tln.modify(tags = [f'ROUND_{d}', f'I{where[0][0]},{where[0][1]}', f'X{where[0][0]}', f'Y{where[0][1]}'])
                        trn.modify(tags = [f'ROUND_{d}', f'I{where[1][0]},{where[1][1]}', f'X{where[1][0]}', f'Y{where[1][1]}'])
                        inverse_peps = (inverse_peps | tln | trn)
                        inverse_peps.view_like_(amp)

                amp_double_layer = (init_peps | inverse_peps)
                amp_double_layer.view_as_(
                    qtn.PEPS,
                    site_tag_id="I{},{}",
                    x_tag_id="X{}",
                    y_tag_id="Y{}",
                    Lx=inverse_peps.Lx,
                    Ly=inverse_peps.Ly,
                    site_ind_id="k{},{}",
                )
                # amp_val = amp_double_layer.contract()
                amp_val = amp_double_layer.contract_boundary_from(
                    from_which='xmin',
                    max_bond=self.max_bond_final,
                    cutoff=0.0,
                    yrange=(0, amp_double_layer.Ly - 1),
                    xrange=(0, amp_double_layer.Lx - 1),
                    mode='mps',
                ).contract()
        return amp_val

    def vamp(self, x, params):
        params = qu.utils.tree_unflatten(params, self.params_pytree)
        return self._vamp(x, params)

    def forward(self, x):
        return self.vamp(x, self.params)


# Circuit TNF classes — canonical home:
# projects/circuitTNF/circuitTNF/VMC/boson/vmap/models.py
# Re-exported here for backward compatibility.
import sys as _sys
_sys.path.insert(0, '/home/sijingdu/TNVMC/projects/circuitTNF')
from circuitTNF.VMC.boson.models import (  # noqa: F401, E402
    PEPS_TNF, MPS_TNF, circuit_TNF, circuit_TNF_2d, circuit_TNF_2d_SU,
)
