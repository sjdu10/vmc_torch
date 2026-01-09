import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ["OMP_NUM_THREADS"] = '1'
from mpi4py import MPI
import numpy as np
import symmray as sr
import quimb.tensor as qtn
import pickle
from autoray import do
from functools import partial
import torch
import json
import time
from tqdm import tqdm
from vmap_utils import sample_next, evaluate_energy, compute_grads, random_initial_config, compute_grads_decoupled
from vmap_utils import NN_fPEPS_Model, fPEPS_Model, Transformer_fPEPS_Model, Transformer_fPEPS_Model_batchedAttn
from vmc_torch.hamiltonian_torch import spinful_Fermi_Hubbard_square_lattice_torch
from vmc_torch.experiment.tn_model import init_weights_to_zero

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

# torch.set_default_device("cuda:0") # GPU
torch.set_default_device("cpu") # CPU
torch.random.manual_seed(42 + RANK)

Lx = 4
Ly = 4
N_f = Lx * Ly - 2 # filling
D = 4
chi = -1
seed = RANK + 42
# only the flat backend is compatible with jax.jit
flat = True
pwd = '/home/sijingdu/TNVMC/VMC_code/vmc_torch/vmc_torch/experiment/vmap/data'
u1z2 = True
appendix = '_U1SU' if u1z2 else ''
params = pickle.load(open(pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/peps_su_params{appendix}.pkl', 'rb'))
skeleton = pickle.load(open(pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/peps_skeleton{appendix}.pkl', 'rb'))
peps = qtn.unpack(params, skeleton)
nsites = peps.nsites
for ts in peps.tensors:
    # print(ts.data)
    ts.modify(data=ts.data.to_flat()*10)
for site in peps.sites:
    peps[site].data._label = site
    peps[site].data.indices[-1]._linearmap = ((0, 0), (1, 0), (1, 1), (0, 1)) # Important for U1->Z2 fPEPS

# Select fPEPS-based model

fpeps_model = Transformer_fPEPS_Model_batchedAttn(
    tn=peps,
    max_bond=chi,
    embed_dim=16,
    attn_heads=4,
    nn_hidden_dim=4*peps.nsites,
    nn_eta=1,
    dtype=torch.float64,
)

model_params_vec = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())
init_std = float(model_params_vec.std())*0.1 # 5e-3
fpeps_model.apply(lambda x: init_weights_to_zero(x, std=init_std))

# fpeps_model = fPEPS_Model(
#     peps, max_bond=chi, dtype=torch.float64
# )

n_params = sum(p.numel() for p in fpeps_model.parameters())
if RANK == 0:
    # print model size
    print(f'fPEPS-based model number of parameters: {n_params}')

# Hamiltonian definition
t=1.0
U=8.0
n_fermions_per_spin = (N_f // 2, N_f // 2)
H = spinful_Fermi_Hubbard_square_lattice_torch(
    Lx,
    Ly,
    t,
    U,
    N_f,
    pbc=False,
    n_fermions_per_spin=n_fermions_per_spin,
    no_u1_symmetry=False,
)
graph = H.graph

if Lx*Ly <= 6 and RANK == 0:
    H_dense = torch.tensor(H.to_dense())
    psi_vec = fpeps_model(torch.tensor(H.hilbert.all_states(), dtype=torch.int32))
    energies_exact, states_exact = torch.linalg.eigh(H_dense)
    print(f'Exact ground state energy: {energies_exact[0].item()/nsites}')
    SU_E = (psi_vec.conj().T @ H_dense @ psi_vec) / (psi_vec.conj().T @ psi_vec)
    print(f'SU variational energy: {SU_E.item()/nsites}')

    terms = sr.hamiltonians.ham_fermi_hubbard_from_edges(
        "Z2",
        edges=tuple(peps.gen_bond_coos()),
        U=8,
        mu=0.0,
    )
    terms = {k: v.to_flat() for k, v in terms.items()}
    new_peps = peps.copy()
    new_peps.apply_to_arrays(lambda x: np.array(x))
    E_double = new_peps.compute_local_expectation_exact(terms, normalized=True)
    print(f'Double layer energy: {E_double/nsites}')



# Total sample size
Ns = int(2e3) 
# batchsize per rank
B = 256
B_grad = 64
# Choose gradient computation method
tn_nn_decouple = False
if tn_nn_decouple:
    get_grads = partial(compute_grads_decoupled, verbose=True, batch_size=B_grad)  # use the decoupled gradient computation
else:
    get_grads = partial(compute_grads, vectorize=True, vmap_grad=True, batch_size=B_grad, verbose=True)  # default to use the vectorized gradient computation

# initial samples for each rank
fxs = []
for _ in range(B):
    fxs.append(random_initial_config(N_f, nsites))
fxs = torch.stack(fxs)
fxs = fxs.to(torch.long)
# burn-in for each rank
t0 = MPI.Wtime()
for _ in range(1):
    fxs, current_amps = sample_next(fxs, fpeps_model, graph)
t1 = MPI.Wtime()
if RANK == 0:
    print(f'Burn-in sampling time: {t1-t0:.4f} s')

vmc_steps = 50
TAG_OFFSET = 424242
vmc_pbar = tqdm(total=vmc_steps, desc="VMC steps")
minSR=False
learning_rate = 0.1

stats_file = pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/vmc_mpi_stats_{fpeps_model._get_name()+str(chi)}.json'
stats = {
    'Np': n_params,
    'sample size': Ns,
    'mean': [],
    'error': [],
    'variance': [],
}
save_state_every = 10

for _ in range(vmc_steps):
    sample_time = 0
    local_energy_time = 0
    grad_time = 0
    t0 = MPI.Wtime()
    message_tag = _
    # rank 0 is the master process, receives data and send out signal for stopping
    
    E_loc_vec = []
    amps_vec = []
    grads_vec_list = []

    n = 0
    n_total = 0
    # terminate = False
    terminate = np.array([0], dtype=np.int32)
    if RANK == 0:
        pbar = tqdm(total=Ns, desc="Sampling starts...")
        fxs, current_amps = sample_next(fxs, fpeps_model, graph)
        energy, local_energies = evaluate_energy(fxs, fpeps_model, H, current_amps)
        grads_vec, amps = get_grads(fxs, fpeps_model)
        
        E_loc_vec.append(local_energies.detach().numpy())
        amps_vec.append(amps.detach().numpy())
        grads_vec_list.append(grads_vec.detach().numpy())

        n += fxs.shape[0]
        n_total += fxs.shape[0]
        pbar.update(fxs.shape[0])

        while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET - 1):
            redundant_message = np.empty(1, dtype=np.int32)
            COMM.Recv(
                [redundant_message, MPI.INT],
                source=MPI.ANY_SOURCE,
                tag=message_tag + TAG_OFFSET - 1,
            )
            # redundant_message = COMM.recv(source=MPI.ANY_SOURCE, tag=message_tag+TAG_OFFSET-1)
            del redundant_message
        
        while not terminate[0]:
            # Receive the local sample count from each rank
            buf = np.empty(1, dtype=np.int32)
            COMM.Recv(
                [buf, MPI.INT], source=MPI.ANY_SOURCE, tag=message_tag + TAG_OFFSET
            )
            dest_rank = buf[0]
            n_total += fxs.shape[0] # all rank have the same batch size
            pbar.update(fxs.shape[0])
            # Check if we have enough samples
            if n_total >= Ns:
                terminate = np.array([1], dtype=np.int32)
                for dest_rank in range(1, SIZE):
                    COMM.Send(
                        [terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1
                    )
            # Send the termination signal to the rank
            COMM.Send([terminate, MPI.INT], dest=dest_rank, tag=message_tag + 1)
        pbar.close()
    
    else:
        while not terminate[0]:
            t00 = MPI.Wtime()
            fxs, current_amps = sample_next(fxs, fpeps_model, graph, verbose=False)
            t11 = MPI.Wtime()
            energy, local_energies = evaluate_energy(fxs, fpeps_model, H, current_amps, verbose=False)
            t22 = MPI.Wtime()
            grads_vec, amps = get_grads(fxs, fpeps_model)
            t33 = MPI.Wtime()
            sample_time += t11 - t00
            local_energy_time += t22 - t11
            grad_time += t33 - t22

            E_loc_vec.append(local_energies.detach().numpy())
            amps_vec.append(amps.detach().numpy())
            grads_vec_list.append(grads_vec.detach().numpy())

            n += fxs.shape[0] # local number of samples

            # send a signal to rank 0 that we have new samples
            buf = np.array([RANK], dtype=np.int32)
            COMM.Send([buf, MPI.INT], dest=0, tag=message_tag + TAG_OFFSET)
            terminate = np.empty(1, dtype=np.int32)
            COMM.Recv([terminate, MPI.INT], source=0, tag=message_tag + 1)
    
    COMM.Barrier()  
    if RANK == 1:
        print(f'Rank {RANK} B={B}, n_sample={n}\nSampling time: {sample_time:.4f} s, local energy time: {local_energy_time:.4f} s, grad time: {grad_time:.4f} s')

    local_energies = np.concatenate(E_loc_vec)
    grads_vec = np.concatenate(grads_vec_list)
    amps = np.concatenate(amps_vec)

    # use MPI to gather energies and grads from all ranks
    all_energies = COMM.allgather(local_energies)
    all_energies = np.concatenate(all_energies)
    energy = np.mean(all_energies)
    energy_var = np.var(all_energies) / all_energies.shape[0]

    if RANK == 0:
        print(f'\n\nSTEP {_} VMC energy: {energy/nsites}')
        N_total = all_energies.shape[0]
        print(f'Total sample size: {N_total}')

    # SR to compute parameter update
    if minSR:
        all_grads = COMM.gather(grads_vec, root=0) # shape (N_total, Np)
        all_amps = COMM.gather(amps, root=0)
        if RANK == 0:
            all_grads = np.concatenate(all_grads)
            all_amps = np.concatenate(all_amps)
            all_energies = torch.tensor(all_energies, dtype=torch.float64)
            grads_vec = torch.tensor(all_grads, dtype=torch.float64)
            amps = torch.tensor(all_amps, dtype=torch.float64)
            # Now that we have local energies, amps and per-sample gradients, we can compute the energy gradient
            # With the energy gradient, we can further do SR for optimization
            # Or we can do minSR, which is simpler here
            t0_sr = time.time()
            with torch.no_grad():
                all_energies_mean = torch.mean(all_energies)
                # compute log-derivative grads
                all_logamp_grads_vec = grads_vec / amps  # shape (B, Np)
                log_grads_vec_mean = torch.mean(all_logamp_grads_vec, dim=0)  # shape (Np,)

                O_sk = (all_logamp_grads_vec - log_grads_vec_mean[None, :]) / (N_total**0.5)  # shape (N_total, Np)
                T = (O_sk @ O_sk.T.conj())  # shape (N_total, N_total)
                diag_shift = 1e-4
                T += diag_shift * torch.eye(N_total, device=T.device, dtype=T.dtype)
                E_s = (all_energies - all_energies_mean) / (N_total**0.5)  # shape (N_total,)

                # minSR: need to solve O_sk * dp = E_s in the least square sense, using the pseudo-inverse of O_sk to get the minimum norm solution
                T_inv = torch.linalg.pinv(T,  rtol=1e-12, atol=0, hermitian=True)
                dp = O_sk.conj().T @ (T_inv @ E_s)  # shape (Np,)
        
    else:
        # SR with iterative minres solver
        local_logamp_grads_vec = grads_vec / amps  # shape (n, Np)
        local_logamp_grads_vec_sum = np.sum(local_logamp_grads_vec, axis=0)  # shape (Np,)
        local_E_logamp_grads_vec_sum = np.dot(local_energies, local_logamp_grads_vec)  # shape (Np,)
        n_local = local_energies.shape[0]
        N_total = COMM.allreduce(n_local, op=MPI.SUM)

        logamp_grads_vec_sum = COMM.allgather(local_logamp_grads_vec_sum)
        E_logamp_grads_vec_sum = COMM.allgather(local_E_logamp_grads_vec_sum)

        logamp_grads_vec_sum = np.array(logamp_grads_vec_sum)  # shape (SIZE, Np)
        E_logamp_grads_vec_sum = np.array(E_logamp_grads_vec_sum)

        logamp_grads_vec_mean = np.sum(logamp_grads_vec_sum, axis=0) / N_total # shape (Np,)
        E_logamp_grads_vec_mean = np.sum(E_logamp_grads_vec_sum, axis=0) / N_total  # shape (Np,)
        
        
        energy_grad = E_logamp_grads_vec_mean - energy * logamp_grads_vec_mean  # shape (Np,)
        
        def R_dot_x(x, eta=1e-6):
            x_out_local = np.zeros_like(x)
            # use matrix multiplication for speedup
            x_out_local = do('dot', local_logamp_grads_vec.T, do('dot', local_logamp_grads_vec, x))
            # synchronize the result
            x_out = COMM.allreduce(x_out_local, op=MPI.SUM)/N_total
            x_out -= do('dot', logamp_grads_vec_mean, x)*logamp_grads_vec_mean
            return x_out + eta*x
        
        import scipy.sparse.linalg as spla
        def matvec(x):
            return R_dot_x(x, 1e-4)
        A = spla.LinearOperator((n_params, n_params), matvec=matvec)
        b = energy_grad
        dp, info = spla.minres(A, b, rtol=1e-4, maxiter=100)

    if RANK == 0:
        # update params
        params_vec = torch.nn.utils.parameters_to_vector(fpeps_model.parameters())

        new_params_vec = params_vec - learning_rate * torch.tensor(dp, dtype=torch.float64)
    
    COMM.Barrier()
    
    # broadcast the new params to all ranks
    new_params_vec = COMM.bcast(new_params_vec if RANK == 0 else None, root=0)
    # print(f'Rank {RANK} received new params vector of shape: {new_params_vec.shape}')

    # load the new params back to the model
    torch.nn.utils.vector_to_parameters(new_params_vec, fpeps_model.parameters())

    vmc_pbar.update(1)
    t1 = MPI.Wtime()
    if RANK == 0:
        # save step, energy, energy variance to a file (if exists, delete and create a new one)
        
        log_file = pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/vmc_mpi_log_{fpeps_model._get_name()+str(chi)}.txt'
        if os.path.exists(log_file) and _ == 0:
            os.remove(log_file)
        with open(log_file, 'a') as f:
            f.write(f'STEP {_}:\nEnergy per site: {energy/nsites}\nEnergy variance square root: {np.sqrt(energy_var)/nsites}\nSample size: {N_total}\nTime elapsed: {t1 - t0} seconds\n\n')
        # print(f'STEP {_}:\nEnergy per site: {energy/nsites}\nEnergy variance square root: {np.sqrt(energy_var)/nsites}\nSample size: {N_total}\nTime elapsed: {t1 - t0} seconds\n\n')
        # save Np, sample size, mean, error, variance to a json file
        stats['mean'].append(energy/nsites)
        stats['error'].append(np.sqrt(energy_var)/nsites)
        stats['variance'].append(energy_var*all_energies.shape[0]/nsites**2)
        stats['sample size'] = N_total

        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        
        # save model checkpoint every few steps
        if (_ + 1) % save_state_every == 0:
            checkpoint_file = pwd+f'/{Lx}x{Ly}/t=1.0_U=8.0/N={N_f}/Z2/D={D}/checkpoint_step_{fpeps_model._get_name()+str(chi)}_{_+1}.pt'
            torch.save(fpeps_model.state_dict(), checkpoint_file)
    

vmc_pbar.close()