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
from Compare import compare

def extract_midpoints(bounds, N, rng):
    start_q = []
    for i in range(N):
        for b in bounds:
            start_q.append(        rng.uniform(  b[0] , b[1]  )         )
    start_q[1::3] = np.sort(start_q[1::3])
    return start_q

def what_u_get(m, t, signal, noise, Nsources, title):
    x = [bursts(t, Nsources, q) for q in m]
    l1, l2, l3 = np.percentile(x, [10,50, 90], axis = 0)
    plt.plot(t, l2, color = 'blue', alpha = 0.8, label = "Esteemed")
    plt.plot(t, signal , color = 'red', alpha = 0.8, label = "True signal")
    #plt.plot(t, signal, color = 'green', alpha = 0.8, label = "True")
    plt.fill_between(t, l1, l3, alpha = 0.3)
    plt.title(title)
    plt.legend()
    plt.show()
    
    
    

rng = np.random.default_rng(1234)
Nsources = 30
bounds = [(2. ,7.), (1., 2.), (0, 2*np.pi)]
start_q = np.array(extract_midpoints(bounds, Nsources, rng))
time = [0,15]

t, signal, real_q, noise = generate_data( bounds,time, Nsources, sampling_frequency=16, sigma_noise=0.8, rng = rng)

plt.plot(t, signal)
plt.show()

H2 = H(1.5*np.eye(Nsources*3))
H1 = H(.5*np.eye(Nsources*3))
def mass_matrix(dim, rng):
    return rng.lognormal(.05,.2, dim)
H2 = PB_H(mass_matrix)


M = CompactBinieriesSignal( signal, time, Nsources, bounds = bounds*Nsources)
nuts =  NUTS(M, H2, dt = 0.001 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.001, rng=rng)
pb = MyNUTS(M, H2, dt = 0.001, rng=rng)
metropolis = metropolis(M, 0.01, rng = rng)
mala = MALA(M, dt = 0.0001, rng=rng)
hmc = HMC(M, H1, L = 5, dt = 0.001, rng = rng)


algs = [ metropolis, mala, hmc,nuts,  mynuts, pb]

#all_m = compare(algs, 21000, start_q,1000, 100)
all_m = [np.load(a.alg_name()+".npy") for a in algs]
'''
for a, m in zip(algs, all_m):
    plt.scatter(real_q[::3], real_q[1::3], color = 'red', marker = 'x')
    l1,l2,l3 = np.percentile(m, [5,50,95], axis = 0)
    lower_error =  l2 - l1.copy()
    upper_error =  l3.copy() - l2
    asymmetric_error_y = np.array(list(zip(lower_error[1::3], upper_error[1::3]))).T
    asymmetric_error_x = np.array(list(zip(lower_error[::3], upper_error[::3]))).T
    plt.errorbar(l2[::3], l2[1::3], yerr=asymmetric_error_y,xerr=asymmetric_error_x, fmt='.' , label = a.alg_name())
    plt.show()
for a, m in zip(algs, all_m):
    what_u_get(m, t, signal, noise, Nsources, a.alg_name())
'''
for a, m in zip(algs, all_m):
    what_u_get(m, t, signal, noise, Nsources, a.alg_name())
    mean = np.percentile(m,50,axis=0)
    for i in range(Nsources):
        sys.corner_plot(m[:,i*3:(i+1)*3], true = real_q[i*3:(i+1)*3])
    plt.scatter(mean[:][1::3],mean[:][0::3], color = 'blue')
    plt.scatter(real_q[:][1::3], real_q[:][::3], marker= 'x', color = 'red')
    plt.show()
