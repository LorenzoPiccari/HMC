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

class Rosenbrock(Model):
        
    def __init__(self, dim, bounds = None):
        
        if bounds is None: self.bounds = [(-3, 3) for i in range(dim)]
        self.names = ['R' + str(i) for i in range(dim)]
        super().__init__(dim, bounds)


    def distribution(self, q):

        s = [- ( 100 * ( (q[i+1]-q[i]**2)**2 ) + (1-q[i]) ** 2 ) for i in range((self.dim-1)) ]
        
        return -np.sum(s)                       

    def gradient(self,q ):
        d = self.dim
        der = np.zeros(len(q))
        
        a = - 400 * q[:d - 1] * ( q[ 1 :] - q[:d-1]**2) - 2* ( 1 - q[:d-1])
        b = 200 * ( q[1 :] - q[:d-1]**2)
        
        
        der[ : d-1] = a.copy() 
        
        der[ 1 :] += b
        
        return der

H = H(np.ones(2)*.1) 
H2 = PB_H(0., .5, 10)
Model = Rosenbrock(2)
m, rj = MyNUTS(Model,Hamiltonian= H, dt = 0.005, rng = np.random.default_rng(1234)).run(10000)
print(sys.IAT(m))
sys.corner_plot(m)
