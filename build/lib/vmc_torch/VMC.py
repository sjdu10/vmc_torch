import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from mpi4py import MPI
from .global_var import DEBUG, set_debug, TIME_PROFILING
import torch
import json
from tqdm import tqdm
from vmc_torch.optimizer import *

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=6,linewidth=100)

class VMC:
    """
    NOTE: At current stage, we consider only a 2D hamiltoniann defined on a square lattice as the object function.
    1. Perform MC sampling from a parameterized probability distribution
    2. Compute the object function and quantities of interest (e.g. energy, gradient, etc.)
    3. Optimize the parameters of the probability distribution with respect to the object function using certain optimization algorithms
    
    """
    def __init__(
        self,
        hamiltonian,
        variational_state,
        optimizer,
        SWO=False,
        beta=None,
        preconditioner=None,
        scheduler=None,
        step_count=0,
        **kwargs,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            variational_state: The variational state for which the hamiltonian must
                be minimised.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """
        self._hamiltonian = hamiltonian # Use NetKet Hamiltonian object!
        
        self._state = variational_state # A variational state object: a torch function + sampler

        self._optimizer = optimizer

        self._preconditioner = preconditioner

        self._scheduler = scheduler

        self.step_count = step_count

        # SWO parameters
        self.SWO = SWO
        self.beta = beta
            
        
    
    @property
    def preconditioner(self):
        """
        The preconditioner used to modify the gradient.

        This is a function with the following signature

        .. code-block:: python

            precondtioner(vstate: VariationalState,
                          grad: PyTree/vector,
                          step: Optional[Scalar] = None)

        Where the first argument is a variational state, the second argument
        is the PyTree/vector of the gradient to precondition and the last optional
        argument is the step, used to change some parameters along the
        optimisation.
        """
        return self._preconditioner
    
    @property
    def optimizer(self):
        """
        The optimizer used to update the parameters of the variational state.
        """
        return self._optimizer
    
    @property
    def scheduler(self):
        """
        The scheduler used to update the learning rate of the optimizer.
        """
        return self._scheduler
    
    @property
    def state(self):
        """
        The variational state of the driver.
        """
        return self._state
    
    @property
    def hamiltonian(self):
        """
        The Hamiltonian of the driver.
        """
        return self._hamiltonian


    def __repr__(self):
        return (
            "Vmc("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self._state})"
        )

    def run(self, start, stop, tmpdir=None, save=True): # Now naive implementation
        """Run the VMC optimization loop."""
        self.Einit = 0.
        MC_energy_stats = {'sample size:': self._state.Ns, 'mean': [], 'error': [], 'variance': []}
        self.step_count = start
        for step in range(start, stop):
            if RANK == 0:
                print('Variational step {}'.format(step))
            self.step_count += 1

            # Update the learning rate if a scheduler is provided
            if self.scheduler is not None:
                learning_rate = self.scheduler(self.step_count)
                self._optimizer.lr = learning_rate

            # Compute the average energy and estimated energy gradient, meanwhile also record the amplitude_grad matrix
            # Use MPI for the sampling
            # Only rank 0 collects the energy statistics, but all ranks must have the energy gradient
            if self._state.equal_partition:
                state_MC_energy, state_MC_loss_grad = self._state.expect_and_grad(self._hamiltonian)
            else:
                state_MC_energy, state_MC_loss_grad = self._state.expect_and_grad(self._hamiltonian, message_tag=step)
            state_MC_loss_grad = COMM.bcast(state_MC_loss_grad, root=0)
            
            MC_energy_stats['mean'].append(state_MC_energy['mean'])
            MC_energy_stats['error'].append(state_MC_energy['error'])
            MC_energy_stats['variance'].append(state_MC_energy['variance'])

            # Precondition the gradient through SR (optional)
            preconditioned_grad = self.preconditioner(self._state, state_MC_loss_grad)

            if RANK == 0:
                print('Energy: {}, Err: {}, \nMAX gi: {}, Max SR gi (optional): {}, Max param: {}\n'.format(
                    state_MC_energy['mean'], 
                    state_MC_energy['error'], 
                    np.max(np.abs(state_MC_loss_grad)), 
                    np.max(np.abs(preconditioned_grad.detach().numpy())), 
                    np.max(np.abs(self._state.params_vec.detach().numpy())))
                )
                # Compute the new parameter vector
                new_param_vec = self._optimizer.compute_update_params(self._state.params_vec, preconditioned_grad) # Subroutine: rank 0 computes new parameter vector based on the gradient
                new_param_vec = new_param_vec.detach().numpy()
                
                self._state.reset() # Clear out the gradient of the state parameters
                
                
                if tmpdir is not None and save:
                    # with open(tmpdir, 'a') as f:
                        # f.write('Variational step {}\n'.format(step))
                        # f.write('Energy: {}, Err: {}\n'.format(state_MC_energy['mean'], state_MC_energy['error']))
                    # save the energy statistics and model parameters to local directory
                    path = tmpdir
                    params_path = path + f'/model_params_step{step}.pth'

                    combined_data = {
                        'model_structure': self._state.model_structure,  # model structure as a dict
                        'model_params_vec': self._state.params_vec.detach().numpy(),  # NumPy array
                        'model_state_dict': self._state.state_dict,  # PyTorch state_dict
                        'MC_energy_stats': state_MC_energy
                    }

                    # Include optimizer state if using SGD with momentum
                    if isinstance(self.optimizer, SGD_momentum):
                        optimizer_state = {
                            'optimizer': 'SGD_momentum',  # Store the optimizer object
                            'velocity': self.optimizer.velocity  # Store the velocity term
                        }
                        combined_data['optimizer_state'] = optimizer_state
                    
                    if isinstance(self.optimizer, Adam):
                        optimizer_state = {
                            'optimizer': 'Adam',  # Store the optimizer object
                            'm': self.optimizer.m,  # Store the m term
                            'v': self.optimizer.v,  # Store the v term
                            't': self.optimizer.t,  # Store the t_step term
                        }
                        combined_data['optimizer_state'] = optimizer_state

                    torch.save(combined_data, params_path)

                    # update the MC_energy_stats.json
                    with open(path + f'/energy_stats_start_{start}.json', 'w') as f:
                        json.dump(MC_energy_stats, f)
                
            else:
                new_param_vec = np.empty(self._state.Np, dtype=float)
                self._state.reset()
            
            self._state.clear_memory() # Clear out the memory

            # Broadcast the new parameter vector to all ranks
            # new_param_vec = np.ascontiguousarray(new_param_vec)
            COMM.Bcast(new_param_vec,root=0)
            # Update the quantum state with the new parameter vector
            self._state.update_state(new_param_vec) # Reload the new parameter vector into the quantum state
            
        return MC_energy_stats
    

    
    def run_SWO(self, start, stop, SWO_max_iter=int(1e3), log_fidelity_tol=1e-4, tmpdir=None, save=True):
        """Run SWO for ground state calculation."""
        assert self.SWO, "SWO is not enabled!"
        self.Einit = 0.
        MC_energy_stats = {'sample size:': self._state.Ns, 'mean': [], 'error': [], 'variance': []}
        self.step_count = start

        # Initialize the SWO parameters
        beta = self.beta

        for t_step in range(start, stop):
            if RANK == 0:
                print('\n\nVariational step {}'.format(t_step))
            self.step_count += 1
            # Step 1: Sample the SWO dataset at the current step for current wavefunction. In this step we use MPI for the sampling.
            # -- Dataset format: [(c, x_c, y_c), ...] where x_c=<c|Psi_(t-1)>, y_c=<c|H|Psi_(t-1)>.
            # -- The dataset is sampled from the current wavefunction |Psi_(t-1)>.
            SWO_dataset = self._state.collect_SWO_dataset_eager(self._hamiltonian, message_tag=t_step)
            # -- Compute energy estimation and record energy statistics
            local_configs = SWO_dataset[0]
            local_configs_amps_dict = SWO_dataset[1]
            local_E_loc_list = []

            for c in local_configs:
                local_E_loc_list.append(local_configs_amps_dict[c][1]/local_configs_amps_dict[c][0])

            E_loc_sum = sum(local_E_loc_list)
            n_total_local = len(local_E_loc_list)
            E_mean_local = E_loc_sum/n_total_local
            n_total = COMM.allreduce(n_total_local, op=MPI.SUM)
            E_mean = COMM.allreduce(E_loc_sum, op=MPI.SUM)/n_total
            local_E_var = np.var(local_E_loc_list, ddof=1)
            local_W_var = (n_total_local-1)*local_E_var + n_total_local*(E_mean_local - E_mean)**2
            E_local_var = COMM.allreduce(local_W_var, op=MPI.SUM)

            # Compute initial fidelity
            local_configs_SWO_amps_dict = {
                c: (local_configs_amps_dict[c][0], (local_configs_amps_dict[c][0]-beta*local_configs_amps_dict[c][1]))
                for c in local_configs
            }
            f1_loc = sum([local_configs_SWO_amps_dict[c][1]/local_configs_SWO_amps_dict[c][0] for c in local_configs])
            f2_loc = sum([local_configs_SWO_amps_dict[c][1]**2/local_configs_SWO_amps_dict[c][0]**2 for c in local_configs])
            f1 = COMM.allreduce(f1_loc, op=MPI.SUM)/n_total
            f2 = COMM.allreduce(f2_loc, op=MPI.SUM)/n_total
            init_fidelity = f1**2/f2

            with torch.no_grad():
                MC_energy_stats['mean'].append(float(E_mean))
                MC_energy_stats['error'].append(float(np.sqrt(E_local_var/n_total**2)))
                MC_energy_stats['variance'].append(float(E_local_var/n_total)) # it is the variance of the mean value of the local energy
                if RANK == 0:
                    print('Energy: {}, Err: {}'.format(E_mean[0], np.sqrt(E_local_var/n_total**2)[0]))
                    print('Initial fidelity: {}, Negative log-fidelity: {}'.format(init_fidelity[0], -np.log(init_fidelity)[0]))
            
            # Step 2: Copy the current wavefunction as the trainable wavefunction for the subsequent SWO iteration.
            # XXX not needed
            ...

            # Step 3: Inner loop: perform supervised learning on the dataset to fit the trainable wavefunction to the target wavefunction
            # -- Inner loop is a normal torch supervised learning loop, with the log-fidelity as the loss function.
            # -- The optimizer can be SGD or Adam, and the scheduler is the learning rate scheduler.

            # Reset the quantum state gradient
            self._state.reset()
            # Add a progress bar for inner loop
            if RANK == 0:
                pbar = tqdm(range(SWO_max_iter))
            # Reset the optimizer
            self._optimizer.reset()

            for SWO_iter in range(SWO_max_iter):
                # Compute the amplitude and the gradient of the amplitude for the training wavefunction
                training_amps_grad_dict = {}
                for c in local_configs:
                    amp, vec_grad = self._state.amplitude_grad(c)
                    training_amps_grad_dict[c] = (amp, vec_grad*amp)
                
                with torch.no_grad():
                    # Compute the fidelity and the gradient of the fidelity using MPI
                    # We can compute the negative log-fidelity of the current training wavefunction from the SWO dataset.
                    # The gradient of the negative log-fidelity can be computed by the gradient of the current training wavefunction.
                    # The gradient can be computed manually. This is inherently a distributed learning task.
                    f1_loc = sum([local_configs_SWO_amps_dict[c][1]/training_amps_grad_dict[c][0] for c in local_configs])
                    f2_loc = sum([local_configs_SWO_amps_dict[c][1]**2/training_amps_grad_dict[c][0]**2 for c in local_configs])
                    f1 = COMM.allreduce(f1_loc, op=MPI.SUM)/n_total
                    f2 = COMM.allreduce(f2_loc, op=MPI.SUM)/n_total
                    fidelity = f1**2/f2
                    log_f = -np.log(fidelity)
                    
                    # Compute the gradient of the fidelity
                    local_loss_grad_1n = -1 * sum(
                            [
                                training_amps_grad_dict[c][1] * local_configs_SWO_amps_dict[c][1]/abs(local_configs_SWO_amps_dict[c][0])**2
                                for c in local_configs
                            ]
                        ) 
                    local_loss_grad_1d = sum(
                            [
                                training_amps_grad_dict[c][0] * local_configs_SWO_amps_dict[c][1]/abs(local_configs_SWO_amps_dict[c][0])**2
                                for c in local_configs
                            ]
                        )
                    local_loss_grad_2n = sum(
                        [
                            training_amps_grad_dict[c][1] * training_amps_grad_dict[c][0]/abs(local_configs_SWO_amps_dict[c][0])**2
                            for c in local_configs
                        ]
                    )
                    local_loss_grad_2d = sum(
                        [
                            abs(training_amps_grad_dict[c][0])**2/abs(local_configs_SWO_amps_dict[c][0])**2
                            for c in local_configs
                        ]
                    )

                loss_grad_1n = COMM.reduce(local_loss_grad_1n, op=MPI.SUM, root=0)
                loss_grad_1d = COMM.reduce(local_loss_grad_1d, op=MPI.SUM, root=0)
                loss_grad_2n = COMM.reduce(local_loss_grad_2n, op=MPI.SUM, root=0)
                loss_grad_2d = COMM.reduce(local_loss_grad_2d, op=MPI.SUM, root=0)

                if RANK == 0:
                    loss_grad_1 = loss_grad_1n/loss_grad_1d
                    loss_grad_2 = loss_grad_2n/loss_grad_2d

                    fidelity_loss_grad = 2*np.real(loss_grad_1 + loss_grad_2)
                    
                    # Update the wavefunction parameters
                    new_param_vec = self._optimizer.compute_update_params(self._state.params_vec, fidelity_loss_grad)
                    new_param_vec = new_param_vec.detach().numpy()
                    pbar.set_description(f'   SWO iter:{SWO_iter}, log-f:{log_f}', refresh=False)
                    pbar.update(1)

                else:
                    new_param_vec = np.empty(self._state.Np, dtype=float)
                
                self._state.clear_memory() # Clear out the memory
                self._state.reset()
                
                # Broadcast the new parameter vector to all ranks
                COMM.Bcast(new_param_vec,root=0)
                # Update the quantum state with the new parameter vector
                self._state.update_state(new_param_vec)

                # Check the convergence of the fidelity
                if log_f < log_fidelity_tol:
                    break

            
            if RANK == 0:
                pbar.close()
                if tmpdir is not None and save:
                    # save the energy statistics and model parameters to local directory
                    path = tmpdir
                    params_path = path + f'/model_params_step{t_step}.pth'

                    combined_data = {
                        'model_structure': self._state.model_structure,  # model structure as a dict
                        'model_params_vec': self._state.params_vec.detach().numpy(),  # NumPy array
                        'model_state_dict': self._state.state_dict,  # PyTorch state_dict
                        'MC_energy_stats': MC_energy_stats
                    }

                    # Include optimizer state if using SGD with momentum
                    if isinstance(self.optimizer, SGD_momentum):
                        optimizer_state = {
                            'optimizer': 'SGD_momentum',  # Store the optimizer object
                            'velocity': self.optimizer.velocity  # Store the velocity term
                        }
                        combined_data['optimizer_state'] = optimizer_state
                    
                    if isinstance(self.optimizer, Adam):
                        optimizer_state = {
                            'optimizer': 'Adam',  # Store the optimizer object
                            'm': self.optimizer.m,  # Store the m term
                            'v': self.optimizer.v,  # Store the v term
                            't': self.optimizer.t,  # Store the t_step term
                        }
                        combined_data['optimizer_state'] = optimizer_state

                    torch.save(combined_data, params_path)

                    # update the MC_energy_stats.json
                    with open(path + f'/energy_stats_start_{start}.json', 'w') as f:
                        json.dump(MC_energy_stats, f)

            
            

            

