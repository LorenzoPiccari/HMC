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
import matplotlib.pyplot as plt
from matplotlib import cm

class Rosenbrock(Model):
        
    def __init__(self, dim, bounds = None):
        
        if bounds is None: self.bounds = [(-3, 3) for i in range(dim)]
        self.names = ['R' + str(i) for i in range(dim)]
        super().__init__(dim, bounds)


    def distribution(self, q):

        s = [ ( 100 * ( (q[i+1]-q[i]**2)**2 ) + ((1-q[i]) ** 2) ) for i in range((self.dim-1)) ]
        
        return np.sum(s)                       

    def gradient(self,q ):
        d = self.dim
        der = np.zeros(len(q))
        
        a = - 400 * q[:d - 1] * ( q[ 1 :] - (q[:d-1]**2)) - 2* ( 1 - q[:d-1])
        b = 200 * ( q[1 :] - (q[:d-1]**2))
        
        
        der[ : d-1] += a
        
        der[ 1 :] += b
        
        return der
    
rng = np.random.default_rng(1234)
dim = 2
M = Rosenbrock(2)
H1 = H(.5*np.eye(dim))
def mass_matrix(dim, rng):
    return .005*np.ones(dim)
    
H2 = PB_H(mass_matrix,100)

start_q = np.array([.2,2.])

nuts =  NUTS(M, H1, dt = 0.001 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.002, rng=rng)
pb = MyNUTS(M, H2, dt = 0.002, rng=rng)
metropolis = metropolis(M, .1, rng = rng)
mala = MALA(M, dt = 0.01, rng=rng)
hmc = HMC(M, H1, L = 20, dt = 0.002, rng = rng)

algs = [metropolis, mala, hmc, nuts, mynuts, pb]
iteration = 5000
'''
for a in algs:
    print(a.alg_name())
    m, rj = a.run(100, start_q)
    m[0,:] = start_q.copy()
    xline = np.linspace(-0.5,2.1,1000)    
    yline = np.linspace(-.1,3.5,1000)   
    
    X,Y = np.meshgrid(xline, yline)
    zline = np.array([np.exp( -M.distribution(np.array([X[i,j],Y[i,j]])) ) for i in range(len(xline)) for j in range(len(yline))])

    plt.style.use('seaborn-whitegrid')
    plt.contour(X, Y, zline.reshape(len(xline),len(yline)), cmap=plt.cm.Greys_r)   

    plt.grid(None)
    plt.plot(m[:,0], m[:,1], c = 'red', marker = '.', alpha=0.5)
    plt.title(a.alg_name())
    plt.show()
'''
m,rj = nuts.run(iteration, start_q)
sys.corner_plot(m,np.array([1,1]))