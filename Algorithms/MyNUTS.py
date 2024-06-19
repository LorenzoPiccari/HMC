
from Algorithms.MC_Algorithm import MC_Algorithm
from Hamiltonian.H import H
import numpy as np
import math
import matplotlib.pyplot as plt
    


class MyNUTS(MC_Algorithm):

    
    def __init__(self, user_model, Hamiltonian = None, dt = 0.005, err = 1., rng = np.random):# dt, metric, integrator
        super().__init__(user_model, rng)
        if Hamiltonian is None:
            self.Hamiltonian = H(np.ones(self.model.dim))
        else: self.Hamiltonian = Hamiltonian
        self.dt = dt
        self.err = err
        
        
    def alg_name(self):
        if isinstance(self.Hamiltonian , H):
            return "MyNUTS"
        else:
            return "PB"

    
    def run(self, iteration, start_q = None):
        print("\nDt:", self.dt)
        return super().run(iteration,start_q)
    
    def run_BMFI(self, iteration, start_q = None):
        
            sample, rejected = self.prep_run(start_q, iteration)
            
            self.E = np.zeros(iteration)
            for i in range(iteration):
                
                sample[i], rj = self.Kernel_BMFI(sample[i-1,:], i)
                
                rejected+= rj
            
            num = 0
            den = 0
            mean = np.mean(self.E)
            for j in range(iteration-1):
                num += (self.E[j+1]- self.E[j])**2
                den += (self.E[j]-mean)**2
            den += (self.E[-1]-mean)**2
            BMFI = num/den    
            return sample, rejected, BMFI
        
    def Kernel_BMFI(self, q, i): # extra_p = time_step, err
        
        q_old = q.copy()
        p = self.Hamiltonian.momentum_update(q, self.model, self.rng)
        self.E[i] = self.Hamiltonian.E_old
        q_l = q.copy()
        q_r = q.copy()
        p_l = p.copy()
        p_r = p.copy()
        j = 1
        n = 1
        stop = 1
        while stop==1:
            v = int(self.rng.uniform()<0.5)*2-1

            if v == 1:
                q_r, p_r, new_q, n2, stop2 = self.build_tree(q_r, p_r, q_l, p_l, q.copy(), v, j)
            else:
                q_l, p_l, new_q, n2, stop2 = self.build_tree(q_r, p_r, q_l, p_l, q.copy(), v, j)

            log_sum_exp = - self.Hamiltonian.E_old + np.log(  n  )
            stop = stop2 * self.Hamiltonian.inversion(q_r, q_l, p_r, p_l)
            
            if n2 > 0:
                n += n2
                log_sum_exp1 = -self.Hamiltonian.E_old + np.log(  n2  )
                if self.rng.uniform() < np.exp(log_sum_exp1 - log_sum_exp):
                        q = new_q.copy()
            
            j +=1
        if (q_old == q).all(): rj = 1
        else: rj = 0
        return q, rj        
    def Kernel(self, q): # extra_p = time_step, err
        
        q_old = q.copy()
        p = self.Hamiltonian.momentum_update(q, self.model, self.rng)
        q_l = q.copy()
        q_r = q.copy()
        p_l = p.copy()
        p_r = p.copy()
        j = 1
        n = 1
        stop = 1
        while stop==1:
            v = int(self.rng.uniform()<0.5)*2-1

            if v == 1:
                q_r, p_r, new_q, n2, stop2 = self.build_tree(q_r, p_r, q_l, p_l, q.copy(), v, j)
            else:
                q_l, p_l, new_q, n2, stop2 = self.build_tree(q_r, p_r, q_l, p_l, q.copy(), v, j)

            log_sum_exp = - self.Hamiltonian.E_old + np.log(  n  )
            stop = stop2 * self.Hamiltonian.inversion(q_r, q_l, p_r, p_l)
            
            if n2 > 0:
                n += n2
                log_sum_exp1 = -self.Hamiltonian.E_old + np.log(  n2  )
                if self.rng.uniform() < np.exp(log_sum_exp1 - log_sum_exp):
                        q = new_q.copy()
            
            j +=1
        if (q_old == q).all(): rj = 1
        else: rj = 0
        return q, rj
    



    def check(self,q ,p, n, picked_q, E_new):
        if math.isnan(E_new):
            return picked_q, n
        n2 = np.exp(-E_new + self.Hamiltonian.E_old)
        if n == 0:
            return q, n2
        else:
            log_sum_exp = -self.Hamiltonian.E_old + np.log(  n  )
            if np.exp(-E_new - log_sum_exp) > self.rng.uniform():
                    picked_q = q.copy()
            
            return picked_q, n+n2
        
        

    def build_tree(self, q_r, p_r, q_l , p_l, picked_q, v, j):

        i=0
        stop = 1
        n = 0 
        
        while i < 2**(j-1) and stop == 1:
            self.dt = v*np.abs(self.dt)
            
            if v == 1:
                    q_r, p_r, n, picked_q, stop = self.step_direction(self.dt, q_r, p_r, q_l, p_l, n, picked_q)
            else:
                    q_l, p_l, n, picked_q, stop = self.step_direction(self.dt, q_l, p_l, q_r, p_r, n, picked_q)
            i+=1
        if v==1:
             return q_r, p_r, picked_q, n, stop
        else:
             return q_l, p_l, picked_q, n, stop
  
    
  
    def step_direction(self, dt, q, p, q_2, p_2,n, picked_q):  
        q, p = self.Hamiltonian.integrator(q.copy(), p.copy(), dt, self.model)
        E_new  = self.Hamiltonian.Energy(q, p, self.model)
        stop = self.Hamiltonian.inversion(q, q_2, p, p_2) * int( ( E_new - self.Hamiltonian.E_old)/self.Hamiltonian.E_old  <= self.err)

        if math.isnan(E_new): 
            
            stop = 0
        if stop:
            q, p = self.model.reflection(q, p)
                
            picked_q, n = self.check(q, p, n, picked_q, E_new)
        return q, p, n, picked_q, stop 
  
    
