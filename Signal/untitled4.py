
from Generate_data import generate_data, bursts
from Signal import Signal
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
import time
from SuperRun import SuperRun, ray_run

def what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, title):
    x = [bursts(t, Nsources_cb, Nsources_bh, q) for q in m]
    l1, l2, l3 = np.percentile(x, [10,50, 90], axis = 0)
    plt.plot(t, l2, color = 'blue', alpha = 0.8, label = "Esteemed")
    plt.plot(t, signal , color = 'red', alpha = 0.8, label = "True signal")
    #plt.plot(t, signal, color = 'green', alpha = 0.8, label = "True")
    plt.title(title)
    plt.fill_between(t, l1, l3, alpha = 0.3)
    plt.legend()
    plt.show()
   
num_process = 2
iteration = 50000
burn_in = 10000
prerun = 100000
times = (0., 50.)  
sigma_noise =.5
rng = np.random.default_rng(1234)
Nsources_cb = 10
bounds_cb  = [(1. ,4.), (1.5, 2.5), (0, 2*np.pi)] 

Nsources_bh = 20
bounds_bh = [(3. ,7.), times, (.5, 1.), (4., 6.), (0, 2*np.pi)]


sampling_freq = 50

t, signal, real_q, noise = generate_data( bounds_cb, bounds_bh, Nsources_cb, Nsources_bh, sampling_frequency=sampling_freq, sigma_noise=sigma_noise, rng = rng)


M = Signal( signal, times,  Nsources_bh, Nsources_cb, sampling_frequency = sampling_freq, sigma_noise= sigma_noise, bounds_cb = bounds_cb*Nsources_cb, bounds_bh = bounds_bh*Nsources_bh)

def mass_matrix(dim, rng):
    return rng.lognormal(1,1, dim)

H1 = H(np.eye(Nsources_bh*5+Nsources_cb*3))
H2 = PB_H(mass_matrix)
    
nuts =  NUTS(M, H1, dt = 0.001 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.00125, rng=rng)
pb = MyNUTS(M, H2, dt = 0.001, rng=rng)
algs = [nuts,mynuts,pb]

for a in algs:
    m = []
    for i in range(2):
        m.append(np.load(a.alg_name()+str(i)+".npy"))
    tot_m = np.concatenate((m[0], m[1]), axis = 0)
    what_u_get(tot_m, t, signal, noise, Nsources_cb, Nsources_bh, a.alg_name())
    mean = np.percentile(tot_m,50,axis=0)
    for i in range(Nsources_cb):
        sys.corner_plot(tot_m[:,i*3:(i+1)*3], true = real_q[i*3:(i+1)*3])
    plt.scatter(mean[:Nsources_cb*3][1::3],mean[:Nsources_cb*3][0::3], color = 'blue')
    plt.scatter(real_q[:Nsources_cb*3][1::3], real_q[:Nsources_cb*3][::3], marker= 'x', color = 'red')
    plt.show()
    for i in range(Nsources_bh):
        sys.corner_plot(tot_m[ :,Nsources_cb*3+ i*5:Nsources_cb*3+(i+1)*5], true = real_q[i*5:(i+1)*5])    
    plt.scatter(mean[Nsources_cb*3:][1::5],mean[Nsources_cb*3:][::5], color = 'blue')
    plt.scatter(real_q[Nsources_cb*3:][1::5], real_q[Nsources_cb*3:][::5], marker= 'x', color = 'red')
    plt.show()
