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

plt.plot(t, signal )
plt.show()

M = Signal( signal, times,  Nsources_bh, Nsources_cb, sampling_frequency = sampling_freq, sigma_noise= sigma_noise, bounds_cb = bounds_cb*Nsources_cb, bounds_bh = bounds_bh*Nsources_bh)

def sort_and_reconstruct(X, N1, N2):
    # Lengths of the vectors
    len3 = 3
    len5 = 5
    
    # Extract vectors of dimension 3 and 5
    vectors_3 = [X[i:i+len3] for i in range(0, N1*len3, len3)]
    vectors_5 = [X[i:i+len5] for i in range(N1*len3, N1*len3 + N2*len5, len5)]
    
    # Sort the vectors by the second element
    vectors_3_sorted = sorted(vectors_3, key=lambda x: x[1])
    vectors_5_sorted = sorted(vectors_5, key=lambda x: x[1])
    
    # Reconstruct the final array
    result = np.concatenate(vectors_3_sorted + vectors_5_sorted)
    
    return result
real_q = sort_and_reconstruct(real_q, Nsources_cb, Nsources_bh)
print(real_q)
def mass_matrix(dim, rng):
    return rng.lognormal(1,1, dim)

H1 = H(np.eye(Nsources_bh*5+Nsources_cb*3))
H2 = PB_H(mass_matrix)

nuts =  NUTS(M, H1, dt = 0.001 , rng=rng)
mynuts = MyNUTS(M, H1, dt = 0.00125, rng=rng)
pb = MyNUTS(M, H2, dt = 0.001, rng=rng)
metropolis = metropolis(M, 0.001, rng = rng)
mala = MALA(M, dt = 1.953125e-05, rng=rng) 
hmc = HMC(M, H1, L = 10, dt = 0.001, rng = rng)

start_q = metropolis.new_point()
algs = [metropolis, mala, hmc, nuts, mynuts, pb]


S_runMynuts = SuperRun(MyNUTS, H1, signal, times, Nsources_bh, Nsources_cb, sampling_freq, sigma_noise, bounds_bh*Nsources_bh, bounds_cb*Nsources_cb)
S_runnuts = SuperRun(NUTS, H1, signal, times, Nsources_bh, Nsources_cb, sampling_freq, sigma_noise, bounds_bh*Nsources_bh, bounds_cb*Nsources_cb)
S_runpb = SuperRun(MyNUTS, H2, signal, times, Nsources_bh, Nsources_cb, sampling_freq, sigma_noise, bounds_bh*Nsources_bh, bounds_cb*Nsources_cb)


m = S_runMynuts.run(iteration, prerun, burn_in)
what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, "bp")
np.save(mynuts.alg_name()+".npy", m)
mean = np.percentile(m,50,axis=0)
print(real_q)
for i in range(Nsources_cb):
    sys.corner_plot(m[:,i*3:(i+1)*3], true = real_q[i*3:(i+1)*3])
plt.scatter(mean[:Nsources_cb*3][1::3],mean[:Nsources_cb*3][0::3], color = 'blue')
plt.scatter(real_q[:Nsources_cb*3][1::3], real_q[:Nsources_cb*3][::3], marker= 'x', color = 'red')
plt.show()
for i in range(Nsources_bh):
    sys.corner_plot(m[ :,Nsources_cb*3+ i*5:Nsources_cb*3+(i+1)*5], true = real_q[i*5:(i+1)*5])    
plt.scatter(mean[Nsources_cb*3:][1::5],mean[Nsources_cb*3:][::5], color = 'blue')
plt.scatter(real_q[Nsources_cb*3:][1::5], real_q[Nsources_cb*3:][::5], marker= 'x', color = 'red')
plt.show()
'''
m = S_runnuts.run(iteration, prerun, burn_in)
what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, "bp")
np.save(nuts.alg_name()+".npy", m)
'''
m = S_runpb.run(iteration, prerun, burn_in)
what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, "bp")
np.save(pb.alg_name()+".npy", m)

'''
m, rj = mala.run(5000, sort_and_reconstruct(start_q, Nsources_cb, Nsources_bh))
mean = np.mean(m, axis = 0)
p = np.percentile(m, 50, axis = 0)
plt.plot(t, bursts(t, Nsources_cb, Nsources_bh, mean) , color = 'green', alpha = 0.8, label = "Mean m")
plt.plot(t, bursts(t, Nsources_cb, Nsources_bh, p) , color = 'orange', alpha = 0.8, label = "P m")

what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, "bp")
#all_m = compare(algs, 21000, start_q, 1000, 100)
#all_m = [np.load(a.alg_name()+".npy") for a in algs]
'''
'''
m = S_run.run(100, 50000, 0)
what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, "bp")

'''
'''
ti=time.time()
m = ray_run(MyNUTS, H1,mynuts.alg_name(), signal, times, Nsources_bh, Nsources_cb, sampling_freq, sigma_noise, bounds_bh*Nsources_bh, bounds_cb*Nsources_cb, iteration, prerun, burn_in, num_process)  
print(time.time()-ti)

for i in range (num_process):
    m = np.load(mynuts.alg_name()+str(i)+".npy")

    what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, "bp")
    
ti=time.time()
m = ray_run(NUTS, H1,nuts.alg_name(), signal, times, Nsources_bh, Nsources_cb, sampling_freq, sigma_noise, bounds_bh*Nsources_bh, bounds_cb*Nsources_cb, iteration, prerun, burn_in, num_process)  
print(time.time()-ti)
for i in range (num_process-1):
    m = np.load(nuts.alg_name()+str(i)+".npy")
    what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, "bp")
    
ti=time.time()
m = ray_run(MyNUTS, H2,pb.alg_name(), signal, times, Nsources_bh, Nsources_cb, sampling_freq, sigma_noise, bounds_bh*Nsources_bh, bounds_cb*Nsources_cb, iteration, prerun, burn_in, num_process)  
print(time.time()-ti)
for i in range (num_process-1):
    m = np.load(pb.alg_name()+str(i)+".npy")
    what_u_get(m, t, signal, noise, Nsources_cb, Nsources_bh, "bp")

#m, rj = mala.run(50000, start_q)
'''
'''
mean = np.mean(m, axis = 0)
p = np.percentile(m, 50, axis = 0)
plt.plot(t, bursts(t, Nsources_cb, Nsources_bh, mean) , color = 'green', alpha = 0.8, label = "True signal")
plt.plot(t, bursts(t, Nsources_cb, Nsources_bh, p) , color = 'orange', alpha = 0.8, label = "True signal")
'''



    
mean = np.percentile(m,50,axis=0)
print(real_q)
for i in range(Nsources_cb):
    sys.corner_plot(m[:,i*3:(i+1)*3], true = real_q[i*3:(i+1)*3])
plt.scatter(mean[:Nsources_cb*3][1::3],mean[:Nsources_cb*3][0::3], color = 'blue')
plt.scatter(real_q[:Nsources_cb*3][1::3], real_q[:Nsources_cb*3][::3], marker= 'x', color = 'red')
plt.show()
for i in range(Nsources_bh):
    sys.corner_plot(m[ :,Nsources_cb*3+ i*5:Nsources_cb*3+(i+1)*5], true = real_q[i*5:(i+1)*5])    
plt.scatter(mean[Nsources_cb*3:][1::5],mean[Nsources_cb*3:][::5], color = 'blue')
plt.scatter(real_q[Nsources_cb*3:][1::5], real_q[Nsources_cb*3:][::5], marker= 'x', color = 'red')
plt.show()
