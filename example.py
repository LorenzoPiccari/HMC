import numpy as np
from Model.Model import Model
import Analysis as sys
from Algorithms.metropolis import metropolis
from Algorithms.MyNUTS import MyNUTS
from Algorithms.NUTS import NUTS
from Algorithms.HMC import HMC
from Algorithms.MALA import MALA
from Hamiltonian.H import H
from Hamiltonian.PB_H import PB_H
from Compare import compare
from scipy.stats import chi2, norm
from scipy import stats

import matplotlib.pyplot as plt

class Rosenbrock(Model):
        
    def __init__(self, dim, bounds = None):
        
        if bounds is None: self.bounds = [(-3, 3) for i in range(dim)]
        self.names = ['R' + str(i) for i in range(dim)]
        super().__init__(dim, bounds)


    def distribution(self, q):
        s = 100*((q[1]-q[0]**2)**2) + (1-q[0])**2
       
        return s                       

    def gradient(self,q ):
        der = np.zeros(len(q))
        
        der[0] = -400*(q[1]-q[0]**2)*q[0] - 2*(1-q[0])
        der[1] = 200*(q[1]-q[0]**2)
        return der
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
    

rng = np.random.default_rng(1234)
dim = 2


M = Rosenbrock(dim)
Mass = np.array([[1,.0], [.0,1]])
#H1 = H(np.ones(dim)*0.01)
H1 = H(Mass*0.001)
def mass_matrix(dim, rng):
    return .0005*np.ones(dim)
    
H2 = PB_H(mass_matrix,10)

start_q = np.ones(dim)

nuts =  NUTS(M, H1, dt = 0.01 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.0001, rng=rng)
pb = MyNUTS(M, H2, dt = 0.0001, rng=rng)
metropolis = metropolis(M, .1, rng = rng)
mala = MALA(M, dt = 0.01, rng=rng)
hmc = HMC(M, H1, L = 40, dt = 0.01, rng = rng)

algs = [metropolis, mala, hmc, nuts, mynuts, pb]
iteration = 10000

m, rj = mynuts.run(iteration, start_q)
sys.corner_plot(m)