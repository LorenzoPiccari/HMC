import numpy as np
from Model.Model import Model
import Analysis as sys
from Algorithms.metropolis import metropolis
from Algorithms.MyNUTS import MyNUTS
from Algorithms.NUTS import NUTS
from Algorithms.HMC import HMC
from Algorithms.MALA import MALA

class Multivariate(Model):
    
    def __init__(self, dim, mean = None, variance = None, bounds = None):
        
        if mean is None: self.mean = np.zeros(dim)
        else: self.mean = mean
        
        if variance is None: self.inv_cov = np.eye(dim)
        else: self.inv_cov = np.linalg.inv(variance)
        
        if bounds is None: bounds = [(self.mean[i]-5, self.mean[i]+5) for i in range(dim)]
        else: bounds = [(bounds[i]-5, bounds[i]+5) for i in range(dim)]
        
        super().__init__(dim, bounds)
        
    def distribution(self, q):
        return 0.5* (q - self.mean) @ (self.inv_cov @ (q - self.mean))


    def gradient(self,q):
        return (self.inv_cov @ (q-self.mean))
    
    
Model = Multivariate(2)
m, rj = MyNUTS(Model, dt = 0.01, rng = np.random.default_rng(1234)).run(10000)
print(sys.IAT(m))
sys.corner_plot(m)
