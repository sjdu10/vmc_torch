import quimb.tensor as qtn
import torch
import torch.nn as nn

from vmc_torch.experiment.tn_model import wavefunctionModel


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
            if not type(x_i) == torch.Tensor:
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
