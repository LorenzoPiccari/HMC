
from Algorithms.MC_Algorithm import MC_Algorithm

import numpy as np

class metropolis(MC_Algorithm):
    
    def __init__(self, user_model, extra_p = .1):
        super().__init__(user_model)
        self.eps = extra_p
    
    def alg_name(self):
            return "metropolis"

        
    def Kernel(self, q):
        new_q = np.random.uniform(q - self.eps, q + self.eps, len(q))

        alpha = self.model.distribution(new_q) - self.model.distribution(q) 
    
        return self.acceptance(q, new_q, alpha)


