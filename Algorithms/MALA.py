
from Algorithms.MC_Algorithm import MC_Algorithm

import numpy as np

class MALA(MC_Algorithm):
    
    
    def __init__(self, user_model, dt = 0.01, jump = 1, rng = np.random ):
        super().__init__(user_model, rng)
        self.dt = dt
        self.jump = jump
    
    
    def alg_name(self):
            return "MALA"

    def Kernel(self, q):
        new_q = q - 0.5*self.dt*self.model.gradient(q) + np.sqrt(self.dt) * self.rng.multivariate_normal(np.zeros(len(q)), self.jump*np.eye(len(q))) 
        
        alpha = self.model.distribution(new_q) - self.model.distribution(q)

    
        return self.acceptance(q, new_q, alpha)


