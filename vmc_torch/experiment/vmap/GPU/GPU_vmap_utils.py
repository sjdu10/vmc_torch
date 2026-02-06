import quimb as qu
import torch
import time
import random
# from mpi4py import MPI
from torch.utils._pytree import tree_map, tree_flatten

#=== Utility functions for Metropolis-Hastings sampling ===#

def propose_exchange_or_hopping(i, j, current_config, hopping_rate=0.25, seed=None):
    if seed is not None:
        random.seed(seed)
    ind_n_map = {0: 0, 1: 1, 2: 1, 3: 2}
    if current_config[i] == current_config[j]:
        return current_config, 0
    proposed_config = current_config.clone()
    config_i = current_config[i].item()
    config_j = current_config[j].item()
    if random.random() < 1 - hopping_rate:
        # exchange
        proposed_config[i] = config_j
        proposed_config[j] = config_i
    else:
        # hopping
        n_i = ind_n_map[current_config[i].item()]
        n_j = ind_n_map[current_config[j].item()]
        delta_n = abs(n_i - n_j)
        if delta_n == 1:
            # consider only valid hopping: (0, u) -> (u, 0); (d, ud) -> (ud, d)
            proposed_config[i] = config_j
            proposed_config[j] = config_i
        elif delta_n == 0:
            # consider only valid hopping: (u, d) -> (0, ud) or (ud, 0)
            choices = [(0, 3), (3, 0)]
            choice = random.choice(choices)
            proposed_config[i] = choice[0]
            proposed_config[j] = choice[1]
        elif delta_n == 2:
            # consider only valid hopping: (0, ud) -> (u, d) or (d, u)
            choices = [(1, 2), (2, 1)]
            choice = random.choice(choices)
            proposed_config[i] = choice[0]
            proposed_config[j] = choice[1]
        else:
            raise ValueError("Invalid configuration")
    return proposed_config, 1


def propose_exchange_or_hopping_vec(i, j, current_configs, hopping_rate=0.25):
    """
    Fully vectorized propose function (GPU Friendly).
    Processes a batch of configurations at once without CPU-GPU synchronization.
    
    Args:
        i, j: (int) site indices for exchange/hopping
        current_configs: (Batch, N_sites) Tensor, dtype=long/int
        hopping_rate: (float) hopping probability
        
    Returns:
        proposed_configs: (Batch, N_sites) new configurations
        change_mask: (Batch,) bool Tensor indicating which samples have valid changes
    """
    B = current_configs.shape[0]
    device = current_configs.device
    
    # Particle number mapping: 0->0, 1->1, 2->1, 3->2
    n_map = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.long)
    
    # Extract column i and j (Batch,)
    col_i = current_configs[:, i]
    col_j = current_configs[:, j]
    
    # 1. Basic check: if both positions have same state, cannot exchange or hop
    diff_mask = (col_i != col_j)
    
    # 2. Random decision between Exchange and Hopping
    rand_vals = torch.rand(B, device=device)
    
    # Only positions with different states need processing
    is_exchange = (rand_vals < (1 - hopping_rate)) & diff_mask
    is_hopping = (~is_exchange) & diff_mask
    
    # Initialize new columns, default equals old ones
    new_col_i = col_i.clone()
    new_col_j = col_j.clone()
    
    # --- A. Handle Exchange (and delta_n=1 Hopping) ---
    # Compute particle numbers
    n_i = n_map[col_i]
    n_j = n_map[col_j]
    delta_n = (n_i - n_j).abs()
    
    # Original logic: simple swap when delta_n == 1
    mask_swap = is_exchange | (is_hopping & (delta_n == 1))
    
    if mask_swap.any():
        new_col_i[mask_swap] = col_j[mask_swap]
        new_col_j[mask_swap] = col_i[mask_swap]
        
    # --- B. Handle Hopping (delta_n = 0 or 2) ---
    
    # Case: delta_n == 0 (e.g. u,d -> 0,ud)
    # Target: randomly become (0, 3) or (3, 0)
    mask_d0 = is_hopping & (delta_n == 0)
    if mask_d0.any():
        rand_bits = torch.randint(0, 2, (B,), device=device, dtype=torch.bool)
        
        val_0 = torch.tensor(0, device=device, dtype=col_i.dtype)
        val_3 = torch.tensor(3, device=device, dtype=col_i.dtype)
        
        # rand=0 -> i=0, j=3; rand=1 -> i=3, j=0
        target_i = torch.where(rand_bits, val_3, val_0)
        target_j = torch.where(rand_bits, val_0, val_3)
        
        new_col_i[mask_d0] = target_i[mask_d0]
        new_col_j[mask_d0] = target_j[mask_d0]

    # Case: delta_n == 2 (e.g. 0,ud -> u,d)
    # Target: randomly become (1, 2) or (2, 1)
    mask_d2 = is_hopping & (delta_n == 2)
    if mask_d2.any():
        rand_bits_2 = torch.randint(0, 2, (B,), device=device, dtype=torch.bool)
        
        val_1 = torch.tensor(1, device=device, dtype=col_i.dtype)
        val_2 = torch.tensor(2, device=device, dtype=col_i.dtype)
        
        # rand=0 -> i=1, j=2; rand=1 -> i=2, j=1
        target_i_2 = torch.where(rand_bits_2, val_2, val_1)
        target_j_2 = torch.where(rand_bits_2, val_1, val_2)
        
        new_col_i[mask_d2] = target_i_2[mask_d2]
        new_col_j[mask_d2] = target_j_2[mask_d2]
        
    # 3. Assemble results
    proposed_configs = current_configs.clone()
    proposed_configs[:, i] = new_col_i
    proposed_configs[:, j] = new_col_j
    
    return proposed_configs, diff_mask


# Batched Metropolis-Hastings updates
@torch.inference_mode()
def sample_next(fxs, fpeps_model, graph, hopping_rate=0.25, verbose=False, **kwargs):
    current_amps = fpeps_model(fxs)
    B = fxs.shape[0]
    device = fxs.device
    
    n_updates = 0 
    
    # Merge row_edges and col_edges loops to reduce duplicate code
    all_edges = []
    for edges in graph.row_edges.values(): 
        all_edges.extend(edges)
    for edges in graph.col_edges.values(): 
        all_edges.extend(edges)

    for edge in all_edges:
        n_updates += 1
        i, j = edge
        
        # Call vectorized function directly without list comprehension
        proposed_fxs, new_flags = propose_exchange_or_hopping_vec(i, j, fxs, hopping_rate)
        
        # Quick check: if all samples have no valid update, skip
        if not new_flags.any():
            continue
        
        # Compute Amplitudes (only for changed ones)
        proposed_amps = current_amps.clone()
        
        # Only compute model for positions where new_flags is True
        new_proposed_fxs = proposed_fxs[new_flags]
        new_proposed_amps = fpeps_model(new_proposed_fxs)
        proposed_amps[new_flags] = new_proposed_amps
        
        # Accept/Reject (fully vectorized, no .item() calls)
        ratio = (proposed_amps.abs()**2) / (current_amps.abs()**2 + 1e-18)
        
        # Vectorized random number generation
        probs = torch.rand(B, device=device)
        
        # Accept mask: only accept if new_flags is True and random < ratio
        accept_mask = new_flags & (probs < ratio)
        
        # Update using masking
        if accept_mask.any():
            fxs[accept_mask] = proposed_fxs[accept_mask]
            current_amps[accept_mask] = proposed_amps[accept_mask]
        
    return fxs, current_amps

@torch.inference_mode()
def evaluate_energy(fxs, fpeps_model, H, current_amps, verbose=False, **kwargs):
    B = fxs.shape[0]
    device = fxs.device
    
    # Prepare connected configurations (Hamiltonian Logic - still on CPU)
    conn_eta_num = []
    conn_etas = []
    conn_eta_coeffs = []
    
    for fx in fxs:
        eta, coeffs = H.get_conn(fx)
        conn_eta_num.append(len(eta))
        conn_etas.append(torch.as_tensor(eta, device=device))
        conn_eta_coeffs.append(torch.as_tensor(coeffs, device=device))
        
    conn_etas = torch.cat(conn_etas, dim=0)
    conn_eta_coeffs = torch.cat(conn_eta_coeffs, dim=0)
    
    # Record number of connected configurations for each sample
    conn_eta_num = torch.tensor(conn_eta_num, device=device)

    # Batch compute connected amplitudes
    chunk_size = B
    conn_amps_list = []
    for i in range(0, conn_etas.shape[0], chunk_size):
        conn_amps_list.append(fpeps_model(conn_etas[i:i+chunk_size]))
    conn_amps = torch.cat(conn_amps_list)

    # Vectorized local energy calculation (eliminate CPU loop)
    batch_ids = torch.repeat_interleave(torch.arange(B, device=device), conn_eta_num)
    
    current_amps_expanded = current_amps[batch_ids]
    
    # Compute each term: H_s's * (psi_s' / psi_s)
    terms = conn_eta_coeffs * (conn_amps / current_amps_expanded)
    
    # Aggregate results
    local_energies = torch.zeros(B, device=device, dtype=terms.dtype)
    local_energies.index_add_(0, batch_ids, terms)

    energy = torch.mean(local_energies)
    
    # if verbose and RANK == 0:
    #     print(f'Energy: {energy.item()}')

    return energy, local_energies

def flatten_params(parameters):
    vec = []
    for param in parameters:
        # Ensure parameters are on the same device
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def compute_grads_gpu(fxs, fpeps_model, vectorize=True, batch_size=None, verbose=False, vmap_grad=False, **kwargs):
    """
    Vectorized gradient computation optimized for GPU with memory chunking and vmap/jacrev support.
    """
    if vectorize:
        # 1. Prepare parameters PyTree structure
        # Compatible with ParameterList, ParameterDict, or direct Tensor List
        params_pytree = (
            list(fpeps_model.params)
            if isinstance(fpeps_model.params, torch.nn.ParameterList)
            else dict(fpeps_model.params)
            if isinstance(fpeps_model.params, torch.nn.ParameterDict)
            else fpeps_model.params
        )

        # ------------------------------------------------------------------
        # Path A: vmap(grad) - Usually efficient for per-sample scalar grad
        # ------------------------------------------------------------------
        if vmap_grad:
            B = fxs.shape[0]
            # Determine chunk size to avoid OOM
            B_grad = batch_size if batch_size is not None else B
            
            # Define logic for a single sample: (x_i, params) -> (amp_i, amp_i)
            def single_sample_amp_func(x_i, p):
                amp = fpeps_model.vamp(x_i.unsqueeze(0), p).squeeze(0)
                return amp, amp  # (Loss target for grad, Aux data for cache)

            # Define vmap-ed gradient function
            # in_dims=(0, None): batch over x, broadcast over params
            grad_vmap_fn = torch.vmap(
                torch.func.grad(single_sample_amp_func, argnums=1, has_aux=True),
                in_dims=(0, None)
            )

            grads_pytree_chunks = []
            amps_chunks = []
            
            t0 = time.time()
            # Chunking loop
            for b_start in range(0, B, B_grad):
                b_end = min(b_start + B_grad, B)
                fxs_chunk = fxs[b_start:b_end]
                
                # Compute gradients and amplitudes for the chunk
                grads_chunk, amps_c = grad_vmap_fn(fxs_chunk, params_pytree)
                
                # IMPORTANT: Detach to free graph memory immediately
                amps_c = amps_c.detach()
                
                grads_pytree_chunks.append(grads_chunk)
                amps_chunks.append(amps_c)
                
                # Explicit delete to help garbage collector
                del grads_chunk, amps_c

            # Aggregate results
            amps = torch.cat(amps_chunks, dim=0)
            # if amps.dim() == 1: amps = amps.unsqueeze(-1)
            
            # Helper to concatenate leaves of the gradient PyTree
            def concat_leaves(*leaves):
                return torch.cat(leaves, dim=0)
            
            # Combine chunks into full batch PyTree
            full_grads_pytree = tree_map(concat_leaves, *grads_pytree_chunks)

            # Flatten to vector (B, Np)
            leaves, _ = tree_flatten(full_grads_pytree)
            # Each leaf is (B, Param_Shape), flatten to (B, Param_Size)
            flat_leaves = [leaf.flatten(start_dim=1) for leaf in leaves]
            batched_grads_vec = torch.cat(flat_leaves, dim=1)
            
            # Final cleanup
            batched_grads_vec = batched_grads_vec.detach()
            fpeps_model.zero_grad()
            
            t1 = time.time()
            if verbose:
                 # Assuming RANK is available globally or passed implicitly
                 try:
                     if RANK == 0: print(f"GPU Batched vmap(grad) time: {t1 - t0:.4f}s")
                 except NameError:
                     print(f"GPU Batched vmap(grad) time: {t1 - t0:.4f}s")

        # ------------------------------------------------------------------
        # Path B: jacrev - Standard Jacobian Reverse Mode
        # ------------------------------------------------------------------
        else:
            def g(x, p):
                results = fpeps_model.vamp(x, p)
                return results, results

            # If no batch_size limit, try to compute all at once (RISKY on GPU)
            if batch_size is None:
                t0 = time.time()
                jac_pytree, amps = torch.func.jacrev(g, argnums=1, has_aux=True)(fxs, params_pytree)
                t1 = time.time()
                if verbose:
                    try: 
                        if RANK == 0: print(f"GPU Full Batch Jacobian time: {t1 - t0:.4f}s")
                    except NameError: pass
            
            # Chunked execution for jacrev
            else:
                B = fxs.shape[0]
                B_grad = batch_size
                jac_pytree_list = []
                amps_list = []
                
                t0 = time.time()
                for b_start in range(0, B, B_grad):
                    b_end = min(b_start + B_grad, B)
                    
                    # jacrev computation
                    jac_pytree_b, amps_b = torch.func.jacrev(g, argnums=1, has_aux=True)(
                        fxs[b_start:b_end], params_pytree
                    )
                    
                    # Detach to save memory
                    amps_b = amps_b.detach()
                    jac_pytree_b = tree_map(lambda x: x.detach(), jac_pytree_b)
                    
                    jac_pytree_list.append(jac_pytree_b)
                    amps_list.append(amps_b)
                    
                    del jac_pytree_b, amps_b

                # Concatenate results
                jac_pytree = tree_map(lambda *leaves: torch.cat(leaves, dim=0), *jac_pytree_list)
                amps = torch.cat(amps_list, dim=0)
                t1 = time.time()
                if verbose:
                    try:
                        if RANK == 0: print(f"GPU Chunked Jacobian time: {t1 - t0:.4f}s")
                    except NameError: pass

            # Process jac_pytree to flat vector
            leaves, _ = tree_flatten(jac_pytree)
            leaves_flattend = [leaf.flatten(start_dim=1) for leaf in leaves]
            batched_grads_vec = torch.cat(leaves_flattend, dim=1)
            
            if amps.dim() == 1: amps.unsqueeze_(1)
            
            # Cleanup
            batched_grads_vec = batched_grads_vec.detach()
            amps = amps.detach()
            if 'jac_pytree' in locals(): del jac_pytree
            fpeps_model.zero_grad()

        # ------------------------------------------------------------------
        # Safety Checks
        # ------------------------------------------------------------------
        if torch.isnan(batched_grads_vec).any() or torch.isinf(batched_grads_vec).any():
            msg = "NaN or Inf detected in batched_grads_vec"
            try:
                print(f"{msg} in RANK {RANK}")
                comm.Abort(1)
            except NameError:
                raise ValueError(msg)
                
        return batched_grads_vec, amps

    else:
        # ------------------------------------------------------------------
        # Non-Vectorized Sequential Fallback
        # ------------------------------------------------------------------
        amps = []
        batched_grads_vec = []
        t0 = time.time()
        
        # Helper to flatten gradients from .grad attributes
        def flatten_grads(model):
            grads_list = []
            for p in model.parameters():
                if p.grad is not None:
                    grads_list.append(p.grad.flatten())
                else:
                    grads_list.append(torch.zeros_like(p).flatten())
            return torch.cat(grads_list)

        for fx in fxs:
            amp = fpeps_model(fx.unsqueeze(0))
            amps.append(amp)
            
            # Standard backward
            amp.backward()
            
            # Collect gradients
            # Assuming params is flat list or consistent iteration order
            current_grads = flatten_grads(fpeps_model)
            batched_grads_vec.append(current_grads)
            
            fpeps_model.zero_grad()
            
        t1 = time.time()
        if verbose:
            try:
                if RANK == 0: print(f"GPU Sequential time: {t1 - t0:.4f}s")
            except NameError: pass
            
        amps = torch.stack(amps, dim=0)
        batched_grads_vec = torch.stack(batched_grads_vec, dim=0)
        
        return batched_grads_vec, amps


def random_initial_config(N_f, N_sites, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    half_filled_config = torch.tensor(
        [1, 2] * (N_sites // 2)
    )
    # Set first (Lx*Ly - N_f) sites to be empty (0)
    empty_sites = list(range(N_sites - N_f))
    doped_config = half_filled_config.clone()
    doped_config[empty_sites] = 0
    # Randomly permute the doped_config
    perm = torch.randperm(N_sites)
    doped_config = doped_config[perm]
    num_1 = torch.sum(doped_config == 1).item()
    num_2 = torch.sum(doped_config == 2).item()
    assert num_1 == N_f // 2 and num_2 == N_f // 2, f"Number of spin up and spin down fermions should be {N_f // 2}, but got {num_1} and {num_2}"

    return doped_config


# =============== Debug ================
def are_pytrees_equal(tree1, tree2):
    from torch.utils import _pytree as pytree
    # Flatten both trees
    leaves1, spec1 = pytree.tree_flatten(tree1)
    leaves2, spec2 = pytree.tree_flatten(tree2)
    
    # 1. Compare structure (TreeSpec)
    if spec1 != spec2:
        print("Tree structures differ.")
        return False
    
    # 2. Compare leaves (Tensors/Values)
    if len(leaves1) != len(leaves2):
        print("Number of leaves differ.")
        return False
        
    for l1, l2 in zip(leaves1, leaves2):
        if torch.is_tensor(l1) and torch.is_tensor(l2):
            if not torch.equal(l1, l2):
                print("Tensor leaves differ.")
                return False
        else:
            if (l1 != l2).any():
                print("Non-tensor leaves differ.")
                print("l1:", l1)
                print("l2:", l2)
                return False
                
    return True
