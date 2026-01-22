import quimb as qu
import torch
import time
import random
from mpi4py import MPI
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

comm = MPI.COMM_WORLD
RANK = comm.Get_rank()
SIZE = comm.Get_size()


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

def propose_exchange_or_hopping_vec(i, j, current_configs, hopping_rate=0.25, **kwargs):
    """
    完全向量化的 Propose 函数 (GPU Friendly)。
    一次性处理一批构型，无 CPU-GPU 同步。
    
    Args:
        i, j: (int) 发生交换/hopping 的 site 索引
        current_configs: (Batch, N_sites) Tensor, dtype=long/int
        hopping_rate: (float) 跳跃概率
        
    Returns:
        proposed_configs: (Batch, N_sites) 新的构型
        change_mask: (Batch,) bool Tensor, 指示哪些样本发生了有效改变
    """
    B = current_configs.shape[0]
    device = current_configs.device
    
    # 粒子数映射表: 0->0, 1->1, 2->1, 3->2
    # 放到 device 上以便索引
    # 建议在外部定义为全局常量以避免重复创建，但在函数内创建开销也很小
    n_map = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.long)
    
    # 提取第 i 列和第 j 列 (Batch,)
    col_i = current_configs[:, i]
    col_j = current_configs[:, j]
    
    # 1. 基础检查：如果两个位置状态相同，则无法交换或hopping
    # 对应原代码: if current_config[i] == current_config[j]: return ..., 0
    diff_mask = (col_i != col_j)
    
    # 2. 随机决定是 Exchange 还是 Hopping
    # 生成 (Batch,) 的随机数
    rand_vals = torch.rand(B, device=device)
    
    # 只有状态不同 (diff_mask) 的才需要处理
    is_exchange = (rand_vals < (1 - hopping_rate)) & diff_mask
    is_hopping = (~is_exchange) & diff_mask
    
    # 初始化新的一列，默认等于旧的
    new_col_i = col_i.clone()
    new_col_j = col_j.clone()
    
    # --- A. 处理 Exchange (以及 delta_n=1 的 Hopping) ---
    # 计算粒子数
    n_i = n_map[col_i]
    n_j = n_map[col_j]
    delta_n = (n_i - n_j).abs()
    
    # 原逻辑：delta_n == 1 时也是简单的 swap
    mask_swap = is_exchange | (is_hopping & (delta_n == 1))
    
    if mask_swap.any():
        new_col_i[mask_swap] = col_j[mask_swap]
        new_col_j[mask_swap] = col_i[mask_swap]
        
    # --- B. 处理 Hopping (delta_n = 0 或 2) ---
    # 仅当 is_hopping 为 True 时才检查这些条件
    
    # Case: delta_n == 0 (e.g. u,d -> 0,ud)
    # 目标: 随机变为 (0, 3) 或 (3, 0)
    mask_d0 = is_hopping & (delta_n == 0)
    if mask_d0.any():
        # 生成随机位: 0 或 1
        rand_bits = torch.randint(0, 2, (B,), device=device, dtype=torch.bool)
        
        # 准备目标值常量
        val_0 = torch.tensor(0, device=device, dtype=col_i.dtype)
        val_3 = torch.tensor(3, device=device, dtype=col_i.dtype)
        
        # 根据 rand_bits 选择 i 是 0 还是 3
        # rand=0 -> i=0, j=3; rand=1 -> i=3, j=0
        target_i = torch.where(rand_bits, val_3, val_0)
        target_j = torch.where(rand_bits, val_0, val_3)
        
        new_col_i[mask_d0] = target_i[mask_d0]
        new_col_j[mask_d0] = target_j[mask_d0]

    # Case: delta_n == 2 (e.g. 0,ud -> u,d)
    # 目标: 随机变为 (1, 2) 或 (2, 1)
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
        
    # 3. 组装结果
    # 克隆整个 configs (GPU上内存拷贝很快)
    proposed_configs = current_configs.clone()
    proposed_configs[:, i] = new_col_i
    proposed_configs[:, j] = new_col_j
    
    # diff_mask 基本上覆盖了所有有效变化，
    # 因为 hopping 规则 (u,d)->(0,3) 肯定会改变状态。
    return proposed_configs, diff_mask


# Batched Metropolis-Hastings updates
@torch.inference_mode()
def sample_next(fxs, fpeps_model, graph, hopping_rate=0.25,verbose=False, seed=None):
    current_amps = fpeps_model(fxs)
    B = len(fxs)
    n = 0
    t0 = time.time()
    for row, edges in graph.row_edges.items():
        for edge in edges:
            n += 1
            # if verbose:
            #     print(f"Processing edge {edge} in row {row}")
            i, j = edge
            proposed_fxs = []
            new_flags = []
            # t0 = time.time()
            fx_id = 0
            for fx in fxs:
                proposed_fx, new = propose_exchange_or_hopping(
                    i,
                    j,
                    fx,
                    hopping_rate=hopping_rate,
                    seed=seed + fx_id if seed is not None else None,
                )
                proposed_fxs.append(proposed_fx)
                new_flags.append(new)
                fx_id += 1
            # t1 = time.time()
            # print(f"Propose time: {t1 - t0}")
            proposed_fxs = torch.stack(proposed_fxs, dim=0)
            if not any(new_flags):
                continue
            # only compute amplitudes for newly proposed configs
            new_proposed_fxs = proposed_fxs[torch.tensor(new_flags, dtype=torch.bool)]
            new_proposed_amps = fpeps_model(new_proposed_fxs)
            # print(f"Number of new proposals: {new_proposed_amps.shape[0]} ({B})" )
            proposed_amps = current_amps.clone()
            proposed_amps[torch.tensor(new_flags, dtype=torch.bool)] = new_proposed_amps
            ratio = proposed_amps**2 / current_amps**2
            accept_prob = torch.minimum(ratio, torch.ones_like(ratio))
            for k in range(B):
                if random.random() < accept_prob[k].item():
                    fxs[k] = proposed_fxs[k]
                    current_amps[k] = proposed_amps[k]

    for col, edges in graph.col_edges.items():
        for edge in edges:
            n += 1
            # if verbose:
            #     print(f"Processing edge {edge} in col {col}")
            i, j = edge
            proposed_fxs = []
            new_flags = []
            fx_id = 0
            for fx in fxs:
                proposed_fx, new = propose_exchange_or_hopping(
                    i,
                    j,
                    fx,
                    hopping_rate=hopping_rate,
                    seed=seed + fx_id if seed is not None else None,
                )
                proposed_fxs.append(proposed_fx)
                new_flags.append(new)
                fx_id += 1
            proposed_fxs = torch.stack(proposed_fxs, dim=0)
            if not any(new_flags):
                continue
            # only compute amplitudes for newly proposed configs
            new_proposed_fxs = proposed_fxs[torch.tensor(new_flags, dtype=torch.bool)]
            new_proposed_amps = fpeps_model(new_proposed_fxs)
            # print(f"Number of new proposals: {new_proposed_amps.shape[0]} ({B})" )
            proposed_amps = current_amps.clone()
            proposed_amps[torch.tensor(new_flags, dtype=torch.bool)] = new_proposed_amps
            ratio = proposed_amps**2 / current_amps**2
            accept_prob = torch.minimum(ratio, torch.ones_like(ratio))
            for k in range(B):
                if random.random() < accept_prob[k].item():
                    fxs[k] = proposed_fxs[k]
                    current_amps[k] = proposed_amps[k]
    t1 = time.time()
    if verbose:
        if RANK == 1:
            print(f"Completed one full sweep of MH updates over {n} edges in time: {t1 - t0}")
    return fxs, current_amps

# Batched Metropolis-Hastings updates with reuse of cached bMPS params
@torch.inference_mode()
def sample_next_reuse(fxs, v_model, graph, hopping_rate=0.25, verbose=False, seed=None, benchmark_model=None):
    B = fxs.shape[0]
    # cache bMPS params along x direction first for reuse
    B_bMPS_params_x_dict, current_amps = v_model.cache_bMPS_params_any_direction_vmap(fxs, direction='x')
    for row, edges in graph.row_edges.items():
        for edge in edges:
            i, j = edge
            proposed_fxs, new_flags = [], []
            fx_id = 0
            for fx in fxs:
                proposed_fx, new = propose_exchange_or_hopping(
                    i,
                    j,
                    fx,
                    hopping_rate=hopping_rate,
                    seed=seed + fx_id if seed is not None else None,
                )
                proposed_fxs.append(proposed_fx)
                new_flags.append(new)
                fx_id += 1
            proposed_fxs = torch.stack(proposed_fxs, dim=0)
            if not any(new_flags):
                if verbose:
                    print(f"No changes proposed in this edge. (x, {row}, edge: {edge})")
                continue
            # compute amplitudes for all proposed configs (because must align batchsize with reused bMPS batchsize)
            new_proposed_fxs = proposed_fxs
            # reuse of bMPS from xmin & xmax w.r.t. row to compute amplitudes
            new_proposed_amps = v_model(
                new_proposed_fxs,
                bMPS_params_x_batched=B_bMPS_params_x_dict,
                bMPS_params_y_batched=None,
                selected_rows=(row,),
                selected_cols=None,
            )
            proposed_amps = new_proposed_amps
            if benchmark_model is not None and verbose:
                benchmark_amps = benchmark_model(new_proposed_fxs)
                print(f"Benchmark vs Reuse amplitudes check (x, {row}):")
                print(torch.allclose(benchmark_amps, new_proposed_amps, atol=1e-5))
            # Metropolis-Hastings acceptance
            ratio = proposed_amps**2 / current_amps**2
            accept_probs = torch.minimum(ratio, torch.ones_like(ratio))
            for k in range(B):
                if random.random() < accept_probs[k].item():
                    fxs[k] = proposed_fxs[k]
                    current_amps[k] = proposed_amps[k]
        
        # update bMPS params to next row for reuse
        if row == v_model.Lx - 1:
            # reach the bottom row, no further bMPS to be updated
            break
        B_bMPS_params_x_dict = v_model.update_bMPS_params_to_row_vmap(
            fxs,
            row_id = row,
            bMPS_params_x_batched=B_bMPS_params_x_dict,
            from_which='xmin',
        )

    B_bMPS_params_y_dict, current_amps = v_model.cache_bMPS_params_any_direction_vmap(fxs, direction='y')
    for col, edges in graph.col_edges.items():
        for edge in edges:
            i, j = edge
            proposed_fxs, new_flags = [], []
            fx_id = 0
            for fx in fxs:
                proposed_fx, new = propose_exchange_or_hopping(
                    i,
                    j,
                    fx,
                    hopping_rate=hopping_rate,
                    seed=seed + fx_id if seed is not None else None,
                )
                proposed_fxs.append(proposed_fx)
                new_flags.append(new)
                fx_id += 1
            proposed_fxs = torch.stack(proposed_fxs, dim=0)
            if not any(new_flags):
                if verbose:
                    print(f"No changes proposed in this edge. (y, {col}, edge: {edge})")
                continue
            # compute amplitudes for all proposed configs (because must align batchsize with reused bMPS batchsize)
            new_proposed_fxs = proposed_fxs
            # reuse of bMPS from ymin & ymax w.r.t. col to compute amplitudes
            new_proposed_amps = v_model(
                new_proposed_fxs,
                bMPS_params_x_batched=None,
                bMPS_params_y_batched=B_bMPS_params_y_dict,
                selected_rows=None,
                selected_cols=(col,),
            )
            if benchmark_model is not None and verbose:
                benchmark_amps = benchmark_model(new_proposed_fxs)
                print(f"Benchmark vs Reuse amplitudes check (y, {col}):")
                print(torch.allclose(benchmark_amps, new_proposed_amps, atol=1e-5))
            proposed_amps = new_proposed_amps
            # Metropolis-Hastings acceptance
            ratio = proposed_amps**2 / current_amps**2
            accept_probs = torch.minimum(ratio, torch.ones_like(ratio))
            for k in range(B):
                if random.random() < accept_probs[k].item():
                    fxs[k] = proposed_fxs[k]
                    current_amps[k] = proposed_amps[k]
        
        # update bMPS params to next col for reuse
        if col == v_model.Ly - 1:
            # reach the rightmost col, no further bMPS to be updated
            break
        B_bMPS_params_y_dict = v_model.update_bMPS_params_to_col_vmap(
            fxs,
            col_id = col,
            bMPS_params_y_batched=B_bMPS_params_y_dict,
            from_which='ymin',
        )

    return fxs, current_amps

# Batched Metropolis-Hastings updates
@torch.inference_mode()
def sample_next_vec(fxs, fpeps_model, graph, hopping_rate=0.25, verbose=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    current_amps = fpeps_model(fxs)
    B = fxs.shape[0]
    device = fxs.device
    
    n_updates = 0
    t0 = time.time()
    
    # 合并 row_edges 和 col_edges 循环，减少重复代码
    all_edges = []
    for edges in graph.row_edges.values(): 
        all_edges.extend(edges)
    for edges in graph.col_edges.values(): 
        all_edges.extend(edges)
    
    # 如果 edge 很多，考虑 shuffle，防止遍历顺序造成的 bias (虽然 sweeps 多了无所谓)
    # random.shuffle(all_edges) 

    for edge in all_edges:
        n_updates += 1
        i, j = edge
        
        # 1. 直接调用向量化函数，不需要 list comprehension
        proposed_fxs, new_flags = propose_exchange_or_hopping_vec(i, j, fxs, hopping_rate)
        
        # 快速检查: 如果所有 sample 都没有产生有效 update (比如都是同色自旋交换)，直接跳过
        if not new_flags.any():
            continue
        
        # 2. Compute Amplitudes (只计算改变了的)
        # 注意: 即使不需要 update 的，proposed_amps 也要初始化为 current_amps
        proposed_amps = current_amps.clone()
        
        # 仅对 new_flags 为 True 的部分计算模型
        new_proposed_fxs = proposed_fxs[new_flags]
        new_proposed_amps = fpeps_model(new_proposed_fxs)
        proposed_amps[new_flags] = new_proposed_amps
        
        # 3. Accept/Reject (完全向量化，无 .item() 调用)
        # ratio calculation
        # 为了数值稳定性，建议在 log domain 做 (如果 model 输出 log_psi)，如果是 psi，直接平方
        # 避免除以 0: 虽然物理上 psi 不应为 0，但可以加个 epsilon
        ratio = (proposed_amps.abs()**2) / (current_amps.abs()**2 + 1e-18)
        
        # 向量化生成随机数
        probs = torch.rand(B, device=device)
        
        # 生成掩码: 只有 new_flags 为 True 且 随机数 < ratio 的才接受
        # torch.minimum(ratio, 1) 其实不需要显式写，因为 probs 也是 [0, 1)
        accept_mask = new_flags & (probs < ratio)
        
        # 4. Update (In-place update using masking)
        # 使用 torch.where 或者索引赋值，避免 Python loop
        if accept_mask.any():
            fxs[accept_mask] = proposed_fxs[accept_mask]
            current_amps[accept_mask] = proposed_amps[accept_mask]

    t1 = time.time()
    if verbose and RANK == 1:
        print(f"Completed one full sweep of MH updates over {n_updates} edges in time: {t1 - t0:.4f}s")
        
    return fxs, current_amps

@torch.inference_mode()
def evaluate_energy(fxs, fpeps_model, H, current_amps, verbose=False):
    # TODO: divide the connected configs into chunks of size fxs.shape[0] to avoid OOM
    B = fxs.shape[0]
    # get connected configurations and coefficients
    conn_eta_num = []
    conn_etas = []
    conn_eta_coeffs = []
    for fx in fxs:
        eta, coeffs = H.get_conn(fx)
        conn_eta_num.append(len(eta))
        conn_etas.append(torch.tensor(eta))
        conn_eta_coeffs.append(torch.tensor(coeffs))

    conn_etas = torch.cat(conn_etas, dim=0)
    conn_eta_coeffs = torch.cat(conn_eta_coeffs, dim=0)

    if verbose:
        if RANK == 1:
            print(f'Prepared batched conn_etas and coeffs: {conn_etas.shape}, {conn_eta_coeffs.shape} (batch size {B})')

    # calculate amplitudes for connected configs, in the future consider TN reuse to speed up calculation, TN reuse is controlled by a param that is not batched over (control flow?)
    conn_amps = torch.cat([fpeps_model(conn_etas[i:i+B]) for i in range(0, conn_etas.shape[0], B)])

    # Local energy \sum_{s'} H_{s,s'} <s'|psi>/<s|psi>

    local_energies = []
    offset = 0
    for b in range(B):
        n_conn = conn_eta_num[b]
        amps_ratio = conn_amps[offset:offset+n_conn] / current_amps[b]
        energy_b = torch.sum(conn_eta_coeffs[offset:offset+n_conn] * amps_ratio)
        local_energies.append(energy_b)
        offset += n_conn
    local_energies = torch.stack(local_energies, dim=0)
    if verbose:
        if RANK == 1:
            print(f'Batched local energies: {local_energies.shape}')

    # Energy: (1/N) * \sum_s <s|H|psi>/<s|psi> = (1/N) * \sum_s \sum_{s'} H_{s,s'} <s'|psi>/<s|psi>
    energy = torch.mean(local_energies)
    if verbose:
        if RANK == 1:
            print(f'Energy: {energy.item()}')

    return energy, local_energies

@torch.inference_mode()
def evaluate_energy_reuse(fxs, v_model, H, current_amps, verbose=False, benchmark_model=None):
    t0 = time.time()
    # Label each connected config with which sample it comes from to enable reuse
    B = fxs.shape[0]
    # cache bMPS params for reuse
    B_bMPS_params_x_dict, B_bMPS_params_y_dict = v_model.cache_bMPS_params_vmap(fxs)

    def detect_changed_row_col_pair(fx1, fx2):
        # currently only support nearest neighbor on square lattice
        Ly = v_model.skeleton._Ly
        changed_pos = torch.nonzero(fx1 - fx2)
        changed_pos_2d = []
        assert changed_pos.shape[0] <= 2, "Expect at most 2 on-site config changes"
        for pos in changed_pos:
            x, y = pos.item() // Ly, pos.item() % Ly
            changed_pos_2d.append( (x, y) )
        if len(changed_pos_2d) == 2:
            delta_row = abs(changed_pos_2d[0][0] - changed_pos_2d[1][0])
            delta_col = abs(changed_pos_2d[0][1] - changed_pos_2d[1][1])
            if delta_row <= delta_col:
                x1 = min(changed_pos_2d, key=lambda t: t[0])[0]
                row = True
                col = False
                return row, col, list(x for x in range(x1, x1+delta_row+1))
            else:
                y1 = min(changed_pos_2d, key=lambda t: t[1])[1]
                row = False
                col = True
                return row, col, list(y for y in range(y1, y1+delta_col+1))
        else:
            row = col = False
            return row, col, None
             
    # --------------------------------------------------------------------------
    # 2. Batched Calculation with Reuse
    # --------------------------------------------------------------------------
    # get connected configurations, coefficients and indices
    conn_eta_num = []
    conn_etas = []
    conn_eta_coeffs = []
    conn_eta_indices = []

    fx_ind = 0
    for fx in fxs:
        eta, coeffs = H.get_conn(fx)
        for i_eta in range(len(eta)):
            r, c, pos = detect_changed_row_col_pair(fx, eta[i_eta])
            conn_eta_indices.append( (fx_ind, i_eta, r, c, pos) )

        conn_eta_num.append(len(eta))
        conn_etas.append(torch.tensor(eta))
        conn_eta_coeffs.append(torch.tensor(coeffs))
        fx_ind += 1

    conn_etas = torch.cat(conn_etas, dim=0)
    conn_eta_coeffs = torch.cat(conn_eta_coeffs, dim=0)

    # group connected configs by changed row/col first
    # within row/col group, further group by position

    # select the indices where r==True and c==False
    # TODO: modify this part, form the pytree with batched leaves to input to reusbale PEPS amplitude calculation [x]
    tasks_map = {} # Key: (mode, indices_tuple), Value: lists of (global_idx, parent_idx)
    
    # Record which are "diagonal terms" (x' == x), these do not need recomputation and can directly reuse current amplitudes
    diagonal_mask = torch.zeros(conn_etas.shape[0], dtype=torch.bool, device=fxs.device)
    diagonal_parent_indices = []
    
    # Iterate over all connected configurations and classify them
    # conn_eta_indices[k] = (parent_fx_ind, i_eta, r_bool, c_bool, pos_list)
    for k, (parent_idx, _, is_row, is_col, indices) in enumerate(conn_eta_indices):
        if indices is None:
            # Config doesn't change (Diagonal term, e.g. density-density interaction)
            diagonal_mask[k] = True
            diagonal_parent_indices.append(parent_idx)
            continue
            
        mode = 'row' if is_row else 'col'
        indices_tuple = tuple(sorted(indices)) 
        group_key = (mode, indices_tuple)
        
        if group_key not in tasks_map:
            tasks_map[group_key] = {'global_idxs': [], 'parent_idxs': []}
        
        tasks_map[group_key]['global_idxs'].append(k)
        tasks_map[group_key]['parent_idxs'].append(parent_idx)

    # --------------------------------------------------------------------------
    # 2. Batched Calculation with Reuse
    # --------------------------------------------------------------------------
    # Preallocate result container
    total_conns = conn_etas.shape[0]
    conn_amps = torch.zeros(total_conns, dtype=current_amps.dtype, device=fxs.device)
    
    # A. Handle diagonal terms (Direct Copy)
    if len(diagonal_parent_indices) > 0:
        # find locations of diagonal terms in conn_etas
        diag_locs = torch.nonzero(diagonal_mask).squeeze()
        # find parent indices
        parents = torch.tensor(diagonal_parent_indices, device=fxs.device)
        # Direct copy: <x|psi>
        conn_amps[diag_locs] = current_amps[parents]

    # B. Handle non-diagonal terms (Grouped Vmap Contraction)
    # Fetch the corresponding Batched Environment Dictionary in the pytree
    def slice_env_dict(env_dict, idxs):
        """
        env_dict: {key: PyTree_of_Tensors}
        idxs: indices to slice the pytree
        
        We need to operate on each value (PyTree) in env_dict,
        using qu.utils.tree_map to go inside the PyTree and slice each leaf Tensor.
        """
        return {
            k: qu.utils.tree_map(lambda x: x[idxs], v) 
            for k, v in env_dict.items()
        }

    for (mode, indices), data in tasks_map.items():
        # get indices within conn_etas
        global_idxs = data['global_idxs']  # positions within conn_etas
        parent_idxs = data['parent_idxs']  # originating from which fxs[i]
        
        # 1. target configs x'
        # Note: conn_etas is a large tensor, slicing directly
        target_configs = conn_etas[global_idxs] # (Batch_Group, N_sites)
        
        # 2. corresponding parent indices (for environment slicing)
        subset_parents = torch.tensor(parent_idxs, device=fxs.device)
        
        # 3. slice environment parameters
        subset_env_x = slice_env_dict(B_bMPS_params_x_dict, subset_parents)
        subset_env_y = slice_env_dict(B_bMPS_params_y_dict, subset_parents)
        
        # 4. reuse forward
        if mode == 'row':
            amps_group = v_model(
                target_configs,
                bMPS_params_x_batched=subset_env_x,
                bMPS_params_y_batched=subset_env_y,
                selected_rows=indices,
                selected_cols=None
            )
        else: # col
            amps_group = v_model(
                target_configs,
                bMPS_params_x_batched=subset_env_x,
                bMPS_params_y_batched=subset_env_y,
                selected_rows=None,
                selected_cols=indices
            )
            
        # 5. fill back results into Tensor
        locs = torch.tensor(global_idxs, device=fxs.device)
        conn_amps[locs] = amps_group

    # --------------------------------------------------------------------------
    # 3. Compute Local Energy
    # --------------------------------------------------------------------------
    # E_loc(x) = \sum_{x'} H_{x,x'} * (psi(x') / psi(x))
    
    # conn_amps is shape: (total_conns,)
    # conn_eta_num[b] tells how many connected configs for config b
    
    local_energies = []
    offset = 0
    
    # simple for loop
    for b in range(B):
        n_conn = conn_eta_num[b]
        
        # get connected amplitudes and coefficients for config b
        # shape: (n_conn,)
        amps_slice = conn_amps[offset : offset + n_conn]
        coeffs_slice = conn_eta_coeffs[offset : offset + n_conn]
        
        # Ratio: psi(x') / psi(x)
        # current_amps[b] is a scalar
        ratio = amps_slice / current_amps[b]
        
        # H_loc = \sum H_{xx'} * Ratio
        e_loc = torch.sum(coeffs_slice * ratio)
        
        local_energies.append(e_loc)
        offset += n_conn
        
    local_energies = torch.stack(local_energies)
    
    # Global Mean Energy
    energy_mean = torch.mean(local_energies)

    # if verbose and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f"Reuse Stats: Processed {len(tasks_map)} groups + diagonal terms.")

    if benchmark_model is not None and verbose:
        t1 = time.time()
        print(f'Energy (Reuse)    : {energy_mean.item()}, time: {t1 - t0:.4f} s')
        # below is logics without reuse implemented yet
        ################################################################################
        current_amps_benchmark = benchmark_model(fxs)
        t0 = time.time()
        # calculate amplitudes for connected configs, in the future consider TN reuse to speed up calculation, TN reuse is controlled by a param that is not batched over (control flow?)
        conn_amps_benchmark = torch.cat([benchmark_model(conn_etas[i:i+B]) for i in range(0, conn_etas.shape[0], B)])

        # Local energy \sum_{s'} H_{s,s'} <s'|psi>/<s|psi>

        local_energies = []
        offset = 0
        for b in range(B):
            n_conn = conn_eta_num[b]
            amps_ratio = conn_amps_benchmark[offset:offset+n_conn] / current_amps_benchmark[b]
            energy_b = torch.sum(conn_eta_coeffs[offset:offset+n_conn] * amps_ratio)
            local_energies.append(energy_b)
            offset += n_conn
        local_energies = torch.stack(local_energies, dim=0)
        # Energy: (1/N) * \sum_s <s|H|psi>/<s|psi> = (1/N) * \sum_s \sum_{s'} H_{s,s'} <s'|psi>/<s|psi>
        energy = torch.mean(local_energies)
        t1 = time.time()
        print(f'Energy (Benchmark): {energy.item()}, time: {t1 - t0:.4f} s')
        ################################################################################

    return energy_mean, local_energies

@torch.inference_mode()
def evaluate_energy_vec(fxs, fpeps_model, H, current_amps, verbose=False):
    B = fxs.shape[0]
    device = fxs.device
    
    # 1. 准备连接配置 (Hamiltonian Logic - 依然在 CPU)
    # 这一步如果 H.get_conn 是瓶颈，需要重写 H 类使其支持 vectorization
    conn_eta_num = []
    conn_etas = []
    conn_eta_coeffs = []
    
    for fx in fxs:
        eta, coeffs = H.get_conn(fx)
        conn_eta_num.append(len(eta))
        conn_etas.append(torch.as_tensor(eta, device=device))     # 直接转到 device
        conn_eta_coeffs.append(torch.as_tensor(coeffs, device=device))
        
    conn_etas = torch.cat(conn_etas, dim=0)
    conn_eta_coeffs = torch.cat(conn_eta_coeffs, dim=0)
    
    # 记录每个 sample 有多少个连接配置，用于后续还原
    conn_eta_num = torch.tensor(conn_eta_num, device=device) # (B,)

    # 2. Batch 计算 connected amplitudes
    # 这是一个大矩阵运算，效率很高
    chunk_size = B # 或者更大，取决于显存
    conn_amps_list = []
    for i in range(0, conn_etas.shape[0], chunk_size):
        conn_amps_list.append(fpeps_model(conn_etas[i:i+chunk_size]))
    conn_amps = torch.cat(conn_amps_list)

    # 3. 向量化计算 Local Energies (消除 CPU Loop)
    
    # 我们需要构造一个 batch_index 来告诉 GPU 哪些 conn_amps 属于第 b 个样本
    # 例如 conn_eta_num = [2, 3], 那么 batch_ids = [0, 0, 1, 1, 1]
    batch_ids = torch.repeat_interleave(torch.arange(B, device=device), conn_eta_num)
    
    # 扩展 current_amps 以匹配 conn_amps 的长度
    # current_amps_expanded = [amp[0], amp[0], amp[1], amp[1], amp[1]]
    current_amps_expanded = current_amps[batch_ids]
    
    # 计算每一项的 H_s's * (psi_s' / psi_s)
    terms = conn_eta_coeffs * (conn_amps / current_amps_expanded)
    
    # 聚合结果: 将 terms 按照 batch_ids 加回到 local_energies
    local_energies = torch.zeros(B, device=device, dtype=terms.dtype)
    local_energies.index_add_(0, batch_ids, terms)

    energy = torch.mean(local_energies)
    
    if verbose and RANK == 1:
        print(f'Energy: {energy.item()}')

    return energy, local_energies

def flatten_params(parameters):
    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def compute_grads(fxs, fpeps_model, vectorize=True, batch_size=None, verbose=False, vmap_grad=False):
    if vectorize:
        # Vectorized gradient calculation
        # need per sample gradient of amplitude -- jacobian
        if vmap_grad:
            B = fxs.shape[0]
            # 确定 chunk size
            B_grad = batch_size if batch_size is not None else B
            
            # 1. 准备参数 PyTree
            # 兼容 ParameterList, ParameterDict 或直接的 Tensor List
            params_pytree = (
                list(fpeps_model.params)
                if isinstance(fpeps_model.params, torch.nn.ParameterList)
                else dict(fpeps_model.params)
                if isinstance(fpeps_model.params, torch.nn.ParameterDict)
                else fpeps_model.params
            )

            # 2. 定义单样本函数 (Single Sample Function)
            # vmap 要求我们定义处理 "单个样本" 的逻辑
            def single_sample_amp_func(x_i, p):
                # x_i: (L,) 来自 vmap 的切片
                # p: 参数 PyTree (Shared)
                
                # 为了复用现有的 vamp (它可能期待 batch 维度), 我们伪造一个 batch=1
                # 并在输出时 squeeze 回 scalar
                
                # 注意：如果你的 fpeps_model.vamp 内部有 vmap，这里会变成嵌套 vmap
                # 但这是被允许的，只是要注意效率。
                # 如果是 Pure TN，建议确保 vamp 能处理 batch=1 的情况
                
                amp = fpeps_model.vamp(x_i.unsqueeze(0), p).squeeze(0)
                return amp, amp # (Loss target, Aux data)

            # 3. 定义 vmap(grad)
            # argnums=1: 对 params_pytree 求导
            # in_dims=(0, None): 对 x_i 进行 batch 映射，对 p 进行广播
            grad_vmap_fn = torch.vmap(
                torch.func.grad(single_sample_amp_func, argnums=1, has_aux=True),
                in_dims=(0, None)
            )

            # 4. Chunking Loop
            grads_pytree_chunks = []
            amps_chunks = []
            
            t0 = time.time()
            for b_start in range(0, B, B_grad):
                b_end = min(b_start + B_grad, B)
                fxs_chunk = fxs[b_start:b_end]
                grads_chunk, amps_c = grad_vmap_fn(fxs_chunk, params_pytree)
                amps_c = amps_c.detach()
                grads_pytree_chunks.append(grads_chunk)
                amps_chunks.append(amps_c)
                
                del grads_chunk, amps_c

            # 5. 结果拼接
            amps = torch.cat(amps_chunks, dim=0)
            if amps.dim() == 1: amps = amps.unsqueeze(-1)
            def concat_leaves(*leaves):
                return torch.cat(leaves, dim=0)
            full_grads_pytree = tree_map(concat_leaves, *grads_pytree_chunks)

            # 6. Flatten to Vector (B, Np)
            leaves, _ = tree_flatten(full_grads_pytree)
            # 每个 leaf 现在是 (B, Param_Shape)
            # Flatten start_dim=1 -> (B, Param_Size)
            flat_leaves = [leaf.flatten(start_dim=1) for leaf in leaves]
            batched_grads_vec = torch.cat(flat_leaves, dim=1)
            
            batched_grads_vec = batched_grads_vec.detach()
            fpeps_model.zero_grad()
            
            t1 = time.time()
            if verbose and RANK == 1:
                print(f"Single Batched vmap(grad) time: {t1 - t0}")

            # detect if batched_grads_vec contains NaN or Inf
            if torch.isnan(batched_grads_vec).any() or torch.isinf(batched_grads_vec).any():
                print(batched_grads_vec[0])
                raise ValueError(f"NaN or Inf detected in batched_grads_vec in RANK {RANK}")
            return batched_grads_vec, amps
        else:
            params_pytree = (
                list(fpeps_model.params)
                if type(fpeps_model.params) is torch.nn.ParameterList
                else dict(fpeps_model.params)
                if type(fpeps_model.params) is torch.nn.ParameterDict
                else fpeps_model.params
            )
            # params is a pytree, fxs has shape (B, nsites)
            def g(x, p):
                results = fpeps_model.vamp(x, p)
                return results, results
            if batch_size is None:
                t0 = time.time()
                jac_pytree, amps = torch.func.jacrev(g, argnums=1, has_aux=True)(fxs, params_pytree)
                t1 = time.time()
                if verbose:
                    if RANK == 1:
                        print(f"Single Batched Jacobian time: {t1 - t0}")
            else:
                B = fxs.shape[0]
                B_grad = batch_size
                jac_pytree_list = []
                amps_list = []
                t0 = time.time()
                for b_start in range(0, B, B_grad):
                    b_end = min(b_start + B_grad, B)
                    jac_pytree_b, amps_b = torch.func.jacrev(g, argnums=1, has_aux=True)(fxs[b_start:b_end], params_pytree)
                    amps_b.detach_()
                    jac_pytree_b = tree_map(lambda x: x.detach(), jac_pytree_b)
                    jac_pytree_list.append(jac_pytree_b)
                    amps_list.append(amps_b)
                # concatenate jac_pytree_list along batch dimension
                jac_pytree = tree_map(lambda *leaves: torch.cat(leaves, dim=0), *jac_pytree_list)
                amps = torch.cat(amps_list, dim=0)
                t1 = time.time()
                if verbose:
                    if RANK == 1:
                        print(f"Single Batched Jacobian time: {t1 - t0}")
            # jac_pytree has shape same as params_pytree, each leaf has shape (B, )

            # Get per-sample batched grads in list of pytree format
            leaves, _ = tree_flatten(jac_pytree) # list of leaves in jac_pytree, each leaf shape (B, param_shape)
            leaves_flattend = [leaf.flatten(start_dim=1) for leaf in leaves]  # each leaf shape (B, param_size)
            batched_grads_vec = torch.cat(leaves_flattend, dim=1) # shape (B, Np), Np is number of parameters
            amps.unsqueeze_(1)  # shape (B, 1)
            
            # clear grads and cached computational graph to save memory on CPU
            batched_grads_vec = batched_grads_vec.detach()
            amps = amps.detach()
            del jac_pytree
            fpeps_model.zero_grad()
            return batched_grads_vec, amps
    
    else:
        # Sequential non-vectorized gradient calculation
        amps = []
        batched_grads_vec = []
        t0 = time.time()
        for fx in fxs:
            amp = fpeps_model(fx.unsqueeze(0))
            amps.append(amp)
            amp.backward()
            grads = qu.tree_map(lambda x: x.grad, list(fpeps_model.params))
            # batched_grads_vec.append(torch.nn.utils.parameters_to_vector(grads))
            batched_grads_vec.append(flatten_params(grads))
            qu.tree_map(lambda x: x.grad.zero_(), list(fpeps_model.params))
        t1 = time.time()
        if verbose and RANK == 1:
            print(f"Single Batched Sequential Jacobian time: {t1 - t0}")
        amps = torch.stack(amps, dim=0)
        batched_grads_vec = torch.stack(batched_grads_vec, dim=0)
        return batched_grads_vec, amps

def compute_grads_decoupled(fxs, fpeps_model, batch_size=None, **kwargs):
    """
    TN-NN decoupled gradient computation:
    Step 1: forward pass to get nn_backflow output (no grad), delta_p
    Step 2: calculate TN gradients (chunked vmap) to get sensitivity vector (dAmp/d(ftn_params) and dAmp/d(delta_p))
    Step 3: propagate sensitivity vector back to NN (VJP, sequential loop)
    """
    B = fxs.shape[0]
    
    B_grad = batch_size if batch_size is not None else B
    
    ftn_params = list(fpeps_model.ftn_params)
    nn_params = list(fpeps_model.nn_backflow.parameters())
    nn_params_dict = dict(zip(fpeps_model.nn_param_names, nn_params))

    with torch.no_grad():
        batch_delta_p = torch.func.functional_call(
            fpeps_model.nn_backflow, nn_params_dict, fxs.long()
        )
    # batch_delta_p shape: (B, ftn_params_length)
    
    def tn_only_func(x_i, ftn_p_list, delta_p_i):
        amp = fpeps_model.tn_contraction(x_i, ftn_p_list, delta_p_i)
        return amp, amp # (Target, Aux)

    tn_grad_vmap_func = torch.vmap(
        torch.func.grad(tn_only_func, argnums=(1, 2), has_aux=True), 
        in_dims=(0, None, 0)
    )

    g_ftn_chunks = []
    g_sensitivity_chunks = []
    amps_chunks = []

    t0 = time.time()
    for b_start in range(0, B, B_grad):
        b_end = min(b_start + B_grad, B)
        
        fxs_chunk = fxs[b_start:b_end]
        delta_p_chunk = batch_delta_p[b_start:b_end]
        
        (g_ftn_c, g_sens_c), amps_c = tn_grad_vmap_func(fxs_chunk, ftn_params, delta_p_chunk)
        
        if amps_c.requires_grad:
            amps_c = amps_c.detach()
            
        g_ftn_chunks.append(g_ftn_c)       
        g_sensitivity_chunks.append(g_sens_c)
        amps_chunks.append(amps_c)
        
        del g_ftn_c, g_sens_c, amps_c
    t1 = time.time()

    g_sensitivity = torch.cat(g_sensitivity_chunks, dim=0)
    
    amps = torch.cat(amps_chunks, dim=0)
    if amps.dim() == 1:
        amps = amps.unsqueeze(-1)

    g_ftn = tree_map(lambda *leaves: torch.cat(leaves, dim=0), *g_ftn_chunks)

    # =================================================================
    g_nn_params_list = []
    t2 = time.time()
    for i in range(B):
        x_i = fxs[i].unsqueeze(0) 
        g_sens_i = g_sensitivity[i].unsqueeze(0) 
        
        fpeps_model.nn_backflow.zero_grad()
        
        with torch.enable_grad():
            out_i = torch.func.functional_call(
                fpeps_model.nn_backflow, 
                nn_params_dict, 
                x_i.long()
            )
            target = torch.sum(out_i * g_sens_i.detach())
            grads_i = torch.autograd.grad(target, nn_params, retain_graph=False)
            
        flat_g = flatten_params(grads_i)
        g_nn_params_list.append(flat_g)
    t3 = time.time()
    if kwargs.get('verbose', True) and RANK == 1:
        print(f"Single Batched grad calc: TN gradient time: {t1 - t0}, Sequential NN VJP time: {t3 - t2}")
    # Stack g_nn_params_list into a tensor of shape (B, Np_nn)
    g_nn_params_vec = torch.stack(g_nn_params_list)


    # Flatten g_ftn
    leaves, _ = tree_flatten(g_ftn)
    flat_g_ftn_list = [leaf.flatten(start_dim=1) for leaf in leaves]
    g_ftn_params_vec = torch.cat(flat_g_ftn_list, dim=1)

    g_params_vec = torch.cat([g_ftn_params_vec, g_nn_params_vec], dim=1) # (B, Np_total)
    
    return g_params_vec, amps

def random_initial_config(N_f, N_sites, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    half_filled_config = torch.tensor(
        [1,2] * (N_sites // 2)
    )
    # set first (Lx*Ly - N_f) sites to be empty (0)
    empty_sites = list(range(N_sites-N_f))
    doped_config = half_filled_config.clone()
    doped_config[empty_sites] = 0
    # randomly permute the doped_config
    perm = torch.randperm(N_sites)
    doped_config = doped_config[perm]
    num_1 = torch.sum(doped_config == 1).item()
    num_2 = torch.sum(doped_config == 2).item()
    assert num_1 == N_f//2 and num_2 == N_f//2, f"Number of spin up and spin down fermions should be {N_f//2}, but got {num_1} and {num_2}"

    return doped_config


# =============== Debug ================
def are_pytrees_equal(tree1, tree2):
    from torch.utils import _pytree as pytree
    import torch
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