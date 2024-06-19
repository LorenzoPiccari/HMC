from Generate_data import generate_data, bursts
from Signal import Signal
from Algorithms.metropolis import metropolis
from Algorithms.MyNUTS import MyNUTS
from Algorithms.NUTS import NUTS
from Algorithms.HMC import HMC
from Algorithms.MALA import MALA
from Hamiltonian.H import H
from Hamiltonian.PB_H import PB_H
import numpy as np
import ray
import matplotlib.pyplot as plt
def what_u_get(x, t, signal, noise, Nsources_cb, Nsources_bh, title):
    plt.plot(t, x, color = 'blue', alpha = 0.8, label = "Esteemed")
    plt.plot(t, signal , color = 'red', alpha = 0.8, label = "True signal")
    #plt.plot(t, signal, color = 'green', alpha = 0.8, label = "True")
    plt.title(title)
    plt.legend()
    plt.show()
    

@ray.remote
def ray_worker(S_run, iteration, preruniteration, burn_in):
    return S_run.run(iteration, preruniteration, burn_in)
    
def ray_run(S_run, iteration, preruniteration, burn_in, num_process):
    
    ray.shutdown()
    ray.init()
    futures = [ray_worker.remote(S_run, iteration, preruniteration, burn_in) for i in range(num_process)]
    results = [ray.get(f) for f in futures]
    ray.shutdown()
        
    return results

class SuperRun:
    def __init__(self,algorithm, hamiltonian,  sample, time,  Nsources_bh, Nsources_cb, sampling_frequency=16, sigma_noise=1,bounds_bh = None, bounds_cb = None, rng = np.random):
        self.algorithm = algorithm
        self.hamiltonian = hamiltonian
        self.rng = rng
        self.model = Signal( sample, time,  Nsources_bh, Nsources_cb, sampling_frequency, sigma_noise,bounds_bh , bounds_cb )
        self.Nsources_bh = Nsources_bh
        self.Nsources_cb = Nsources_cb
    
    def burn_in(self, start_q,preruniteration):
        bad_dt = True
        dt = 0.01
        q = start_q.copy()
        
        while bad_dt:
            m, rj = MALA(self.model, dt).simple_run(100, q)
            print(dt,rj)
            q = m[-1]
            if rj <= 35:
                bad_dt = False
            dt /= 2
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        m, _ = MALA(self.model, dt, rng = self.rng).run(preruniteration, q)
        x = [bursts(self.model.t, self.Nsources_cb, self.Nsources_bh, q) for q in m]
        q1 = np.percentile(x, 50, axis = 0)
        q2 = np.percentile(m, 50, axis=0)
        what_u_get(q1, self.model.t, self.model.sample, self.model.sigma_noise, self.Nsources_cb, self.Nsources_bh, "mid")
        what_u_get(bursts(self.model.t, self.Nsources_cb, self.Nsources_bh, q2), self.model.t, self.model.sample, self.model.sigma_noise, self.Nsources_cb, self.Nsources_bh, "mid")
        
        return q2
        
        
    def run(self, iteration, preruniteration, burn_in=0):
        start_q = MALA(self.model).new_point()
        q = sort_and_reconstruct(start_q, self.Nsources_cb, self.Nsources_bh)

        q = self.burn_in(start_q,preruniteration)
        
        q = sort_and_reconstruct(q, self.Nsources_cb, self.Nsources_bh)
        
        what_u_get(bursts(self.model.t, self.Nsources_cb, self.Nsources_bh, q), self.model.t, self.model.sample, self.model.sigma_noise, self.Nsources_cb, self.Nsources_bh, "mid")
        dt = 0.01
        bad_dt = True
        while bad_dt:
            
            m, rj = self.algorithm(self.model, self.hamiltonian, dt,rng = self.rng).simple_run(100, q)
            q=m[-1]
            print(dt,rj)
            if rj <=25:
                bad_dt = False
            dt /= 2
        '''
        Mass = 1
        bad_mass = True
        while bad_mass:
            hamiltonian = H(Mass*np.eye(len(q)))
            m, rj, BMFI = self.algorithm(self.model, hamiltonian, dt).run_BMFI(100, q)
            print(Mass, BMFI)
            q=m[-1]
            if BMFI >1.10:
                Mass /=2
            if BMFI <.85:    
                Mass *=2
            if BMFI >.85 and BMFI <1.10:
                bad_mass = False
        
        hamiltonian = H(Mass*np.eye(len(q)))  
        '''
        m, rj= self.algorithm(self.model, self.hamiltonian, dt, rng = self.rng).run(iteration, q)
        return m[burn_in:,:], rj
        
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
