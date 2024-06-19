from SignalBHs.GenerateData import generate_data, bursts
from SignalBHs.ModelBHs import BHsSignal
import matplotlib.pyplot as plt
import numpy as np
import Analysis as sys
from Algorithms.metropolis import metropolis
from Algorithms.MyNUTS import MyNUTS
from Algorithms.NUTS import NUTS
from Algorithms.HMC import HMC
from Algorithms.MALA import MALA
from Hamiltonian.H import H
from Hamiltonian.PB_H import PB_H
from Compare import compare

def extract_midpoints(bounds, N, rng):
    start_q = []
    for i in range(N):
        for b in bounds:
            start_q.append(        rng.uniform(  b[0] , b[1]  )         )
    start_q[1::5] = [(i+1)*bounds[1][1]/(N+1) for i in range(N)]
    return start_q

def what_u_get(m, t, signal, noise, Nsources, title):
    x = [bursts(t, Nsources, q) for q in m]
    l1, l2, l3 = np.percentile(x, [10,50, 90], axis = 0)
    plt.plot(t, l2, color = 'blue', alpha = 0.8, label = "Esteemed")
    plt.plot(t, signal , color = 'red', alpha = 0.8, label = "True signal")
    #plt.plot(t, signal, color = 'green', alpha = 0.8, label = "True")
    plt.title(title)
    plt.fill_between(t, l1, l3, alpha = 0.3)
    plt.legend()
    plt.show()
    
    
    

rng = np.random.default_rng(1234)
Nsources = 30
bounds = [(2. ,7.), (0., 15.), (.1, 1.), (.75, 4.), (0, 2*np.pi)]
start_q = np.array(extract_midpoints(bounds, Nsources, rng))

t, signal, real_q, noise = generate_data( bounds, Nsources, sampling_frequency=16, sigma_noise=0.8, rng = rng)

plt.plot(t, signal )
plt.show()
bounds = [(2. ,7.), (0., 15.), (.1, 1.3), (.75, 4.), (0, 2*np.pi)]

M = BHsSignal( signal, Nsources, bounds = bounds*Nsources)
def mass_matrix(dim, rng):
    return rng.lognormal(1,1, dim)
H1 = H(np.eye(Nsources*5))
H2 = PB_H(mass_matrix)

nuts =  NUTS(M, H1, dt = 0.001 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.001, rng=rng)
pb = MyNUTS(M, H2, dt = 0.001, rng=rng)
metropolis = metropolis(M, 0.001, rng = rng)
mala = MALA(M, dt = 0.00001, rng=rng) 
hmc = HMC(M, H1, L = 10, dt = 0.001, rng = rng)

algs = [metropolis, mala, hmc, nuts, mynuts, pb]
'''
m, rj = pb.run(21000, start_q)
np.save(pb.alg_name()+".npy", m[1000:,:])
print(rj)

m, rj = hmc.run(21000, start_q)
np.save(hmc.alg_name()+".npy", m[1000:,:])
print(rj)'''
all_m = compare(algs, 21000, start_q, 1000, 100)
#all_m = [np.load(a.alg_name()+".npy") for a in algs]

for m, a in zip(all_m, algs):
    what_u_get(m, t, signal, noise, Nsources, a.alg_name())


inf= np.load("info.npy")

print(inf)