from SignalCompactBinaries.GenerateData import generate_data, bursts
from SignalCompactBinaries.ModelCompactBinaries import CompactBinieriesSignal
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


def extract_midpoints(bounds, N, rng):
    start_q = []
    for i in range(N):
        for b in bounds:
            start_q.append(        rng.uniform(  b[0] , b[1]  )         )
        
    return start_q

def what_u_get(m, t, signal, noise, Nsources):
    x = [bursts(t, Nsources, q) for q in m]
    l1, l2, l3 = np.percentile(x, [10,50, 90], axis = 0)
    plt.plot(t, l2, color = 'blue', alpha = 0.8, label = "Esteemed")
    plt.plot(t, signal , color = 'red', alpha = 0.8, label = "True signal")
    #plt.plot(t, signal, color = 'green', alpha = 0.8, label = "True")
    plt.fill_between(t, l1, l3, alpha = 0.3)
    plt.legend()
    plt.show()
    
    
    

rng = np.random.default_rng(1234)
Nsources = 5
bounds = [(2. ,7.), (0., 10.), (.1, 1.), (.75, 4.), (0, 2*np.pi)]
start_q = np.array(extract_midpoints(bounds, Nsources, rng))

t, signal, real_q, noise = generate_data( bounds, Nsources, sampling_frequency=16, sigma_noise=0.8, rng = rng)

plt.plot(t, signal + noise)
plt.show()
bounds = [(2. ,7.), (0., 10.), (.1, 1.3), (.75, 4.), (0, 2*np.pi)]

M = CompactBinieriesSignal( signal, Nsources, bounds = bounds*Nsources)

H2 = PB_H(.5, 1., 10.)
m, rj = MyNUTS(M, H2, dt = 0.0003, rng=rng).run(10000, start_q)
what_u_get(m, t, signal, noise, Nsources)


H1 = H(np.ones(Nsources*5))
m, rj = MyNUTS(M, H1, dt = 0.0003, rng=rng).run(10000, start_q)
what_u_get(m, t, signal, noise, Nsources)


m, rj = NUTS(M, H1, dt = 0.0003, rng=rng).run(10000, start_q)
what_u_get(m, t, signal, noise, Nsources)


m, rj = MALA(M, dt = 0.000001, rng=rng).run(10000, start_q)
what_u_get(m, t, signal, noise, Nsources)




