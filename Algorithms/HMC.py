
from Hamiltonian.H import H
from Algorithms.MC_Algorithm import MC_Algorithm
import numpy as np

class HMC(MC_Algorithm):
            
        def __init__(self, user_model, Hamiltonian = None, L = 30, dt = 0.001):
            super().__init__(user_model)
            if Hamiltonian is None:
                self.Hamiltonian = H(np.ones(self.model.dim))
            else:
                self.Hamiltonian = Hamiltonian
            self.L = L
            self.dt = dt
             
                
        def alg_name(self):
            return "HMC"

        def run(self, iteration, start_q = None):
            
            return super().run(iteration,start_q)
            
            
        def Kernel(self, q): #extra_p = (dt, L)
                    
                p = self.Hamiltonian.momentum_update(q, self.model, self.rng)
                
                
                new_q, p = self.L_leap(q.copy(), p.copy())
                
                
                E_new  = self.Hamiltonian.Energy(new_q, p, self.model)
                
                alpha = E_new - self.Hamiltonian.E_old
                
                new , rj = self.acceptance(q, new_q, alpha)
                
                return new, rj



        def L_leap(self, q, p):
                for i in range(self.L):
                        q, p = self.Hamiltonian.integrator(q, p, self.dt, self.model)
                return q, p
