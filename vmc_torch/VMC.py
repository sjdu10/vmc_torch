import numpy as np
from mpi4py import MPI
from .global_var import DEBUG, set_debug
import torch
import json

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=6,linewidth=100)

class VMC:
    """
    NOTE: At current stage, we consider only a 2D hamiltonianiltonian defined on a square lattice as the object function.
    1. Perform MC sampling from a parameterized probability distribution
    2. Compute the object function and quantities of interest (e.g. energy, gradient, etc.)
    3. Optimize the parameters of the probability distribution with respect to the object function using certain optimization algorithms
    
    """
    def __init__(
        self,
        hamiltonian,
        variational_state,
        optimizer,
        preconditioner=None,
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

        self.step_count = step_count
        
    
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
        for step in range(start, stop):
            if RANK == 0:
                print('Variational step {}'.format(step))
            self.step_count += 1
            # Compute the average energy and estimated energy gradient, meanwhile also record the amplitude_grad matrix

            # Use MPI for the sampling
            # Only rank 0 collects the energy statistics, but all ranks must have the energy gradient
            state_MC_energy, state_MC_energy_grad = self._state.expect_and_grad(self._hamiltonian)
            state_MC_energy_grad = COMM.bcast(state_MC_energy_grad, root=0)
            
            MC_energy_stats['mean'].append(state_MC_energy['mean'])
            MC_energy_stats['error'].append(state_MC_energy['error'])
            MC_energy_stats['variance'].append(state_MC_energy['variance'])

            # Precondition the gradient through SR
            preconditioned_grad = self.preconditioner(self._state, state_MC_energy_grad)

            if RANK == 0:
                print('Energy: {}, Err: {}, \nMAX gi: {}, Max SR gi: {}, Max param: {}\n'.format(
                    state_MC_energy['mean'], 
                    state_MC_energy['error'], 
                    np.max(np.abs(state_MC_energy_grad)), 
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
                    torch.save(combined_data, params_path)

                    # update the MC_energy_stats.json
                    with open(path + f'/energy_stats_start_{start}.json', 'w') as f:
                        json.dump(MC_energy_stats, f)
                
            else:
                new_param_vec = np.empty(self._state.Np, dtype=float)

            # Broadcast the new parameter vector to all ranks
            new_param_vec = np.ascontiguousarray(new_param_vec)
            COMM.Bcast(new_param_vec,root=0)
            # Update the quantum state with the new parameter vector
            self._state.update_state(new_param_vec) # Reload the new parameter vector into the quantum state
            
        return MC_energy_stats
    