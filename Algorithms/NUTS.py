from Hamiltonian.H import H
from Algorithms.MC_Algorithm import MC_Algorithm
import numpy as np
import math

class NUTS(MC_Algorithm):

    
    def __init__(self, user_model, Hamiltonian = None, dt = 0.005, err = 1., rng = np.random):# dt, metric, integrator
        super().__init__(user_model, rng)
        if Hamiltonian is None:
            self.Hamiltonian = H(np.ones(self.model.dim))
        else:
            self.Hamiltonian = Hamiltonian
            
        self.dt = dt
        self.err = err
        
        



    def alg_name(self):
        return "NUTS"

    
    def run(self, iteration, start_q = None):
        self.j = 0
        print("\nDt:", self.dt)
        return super().run(iteration,start_q)

            
    def Kernel(self, q): 
        
        q_old = q.copy()
        p = self.Hamiltonian.momentum_update(q, self.model, self.rng)
        
        u = self.rng.exponential(1)
        
        
        q_r, p_r, q_l, p_l, n, stop = self.first_step(q, p, u)
        picked_q = q.copy()
        j = 1
        
        while stop==1:
            
            v = int(self.rng.uniform()<0.5)*2-1
            dt = self.dt * v
            
            if v == 1:
                
                picked_q, q_r, p_r, n2, stop = self.tree(q_r, p_r, q_l, p_l, dt, j, u, picked_q)
            else:
                picked_q, q_l, p_l, n2, stop = self.tree(q_l, p_l, q_r, p_r, dt, j, u, picked_q)
            
            n += n2
            
            if n2/n > self.rng.uniform(0,1):
                    q = picked_q.copy()
            
            j +=1
        
        if (q_old == q).all(): rj = 1
        else: rj = 0
        return q, rj
    

    def first_step(self, q, p, u):
        
        v = int(self.rng.uniform()<0.5)*2-1
        dt = self.dt * v
        if v == 1:
            
            q1, p1 = self.Hamiltonian.integrator(q.copy(), p.copy(), dt, self.model)
            
            E1  = self.Hamiltonian.Energy(q1, p1, self.model)
            if math.isnan(E1):
                return q1, p1, q, p,  1, 0
            return q1, p1, q, p,  1 + int(u >= E1- self.Hamiltonian.E_old), 1
        
        else:
            q1, p1 = self.Hamiltonian.integrator(q.copy(), p.copy(), dt, self.model)

            E1  = self.Hamiltonian.Energy(q1, p1, self.model)
            if math.isnan(E1):
                return q1, p1, q, p,  1, 0
            return q, p, q1, p1, 1 + int(u >= E1- self.Hamiltonian.E_old), 1
    
    def binary_tree(self, q, p, q_dir , p_dir, dt, j, u, picked_q):
        q, p = self.Hamiltonian.integrator(q, p, dt, self.model)
        
        E_1  = self.Hamiltonian.Energy(q, p, self.model)
        if math.isnan(E_1):
            return picked_q, q_dir,p_dir, 0, 0
        n1 = int(u >= E_1 - self.Hamiltonian.E_old )
        
        stop = self.Hamiltonian.inversion(q,q_dir,p,p_dir) * int( ( E_1 - self.Hamiltonian.E_old)/self.Hamiltonian.E_old  <= self.err)
        
        if stop == 1:
            q1, p1 = self.Hamiltonian.integrator(q.copy(), p.copy(), dt, self.model)

            E_2  = self.Hamiltonian.Energy(q, p, self.model)
            if math.isnan(E_2):
                return picked_q, q1,p1, 0, 0
            n2 = int(u >= E_2 - self.Hamiltonian.E_old )
            stop = self.Hamiltonian.inversion(q,q_dir,p,p_dir) * int( ( E_2 - self.Hamiltonian.E_old)/self.Hamiltonian.E_old  <= self.err)
            
            if stop == 1:
                n = n1 + n2
                if n == 0:
                    return picked_q, q1,p1,n,stop
                else:
                    if n1/n > self.rng.uniform(0, 1):
                        return q, q1, p1, n, stop
                    
                    else:
                        return q1, q1, p1, n, stop
        
        return 0, 0, 0, 0, 0
    
    
    def tree(self, q, p, q_dir , p_dir, dt, j, u, picked_q):
        
        if j == 1:
            
            return self.binary_tree( q, p, q_dir , p_dir, dt, j, u, picked_q)
    
  
        else:
            
            picked_q1, q, p, n1, stop = self.binary_tree(q, p, q_dir, p_dir, dt, j-1, u, picked_q)
            
            if stop == 1:
                picked_q2, q, p, n2, stop = self.binary_tree(q, p, q_dir, p_dir, dt, j-1, u, picked_q)
                
                if stop == 1:    
                    n = n1 + n2
                    if n == 0:
                        return picked_q, q, p, n, stop
                    else:
                        if n1/n > self.rng.uniform(0,1):
                            return picked_q1, q, p, n, stop
                        else:
                            return picked_q2, q, p, n, stop
            return 0, 0, 0, 0, 0
            