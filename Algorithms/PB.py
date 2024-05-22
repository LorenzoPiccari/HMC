from Algorithms.MC_Algorithm import MC_Algorithm
from Hamiltonian.PB_H import PB_H
import numpy as np



    


class PB(MC_Algorithm):

    
    def __init__(self, user_model, Hamiltonian = None, dt = 0.005, err = .2):# dt, metric, integrator
        super().__init__(user_model)
        if Hamiltonian is None:
            self.Hamiltonian = PB_H(0., 1.)
        else:
            self.Hamiltonian = Hamiltonian
            
        self.dt = dt
        self.err = err
        
        
    def alg_name(self):
        return "PB"

    
    def run(self, iteration, start_q = None):
        print("\nDt:", self.dt)
        return super().run(iteration,start_q)

            
    def Kernel(self, q): # extra_p = time_step, err
        
        q_old = q.copy()
        p = self.Hamiltonian.momentum_update(q, self.model)
        q_l = q.copy()
        q_r = q.copy()
        p_l = p.copy()
        p_r = p.copy()
        j = 1
        n = 1
        stop = 1
        while stop==1:
            
            v = int(np.random.uniform()<0.5)*2-1

            if v == 1:
                q_r, p_r, new_q, n2, stop2 = self.build_tree(q_r, p_r, q_l, p_l, q, v, j)
            else:
                q_l, p_l, new_q, n2, stop2 = self.build_tree(q_r, p_r, q_l, p_l, q, v, j)
            
            n += n2
            log_sum_exp = -self.Hamiltonian.E_old + np.log(  n + n2 )
            #stop = stop2 *  self.inversion2(p_r, p_l)
            stop = stop2 * self.Hamiltonian.inversion(q_r, q_l, p_r, p_l)
            log_sum_exp1 = -self.Hamiltonian.E_old + np.log(  n )
            
            if stop == 1 and np.random.uniform() < np.exp(log_sum_exp1 - log_sum_exp):
                    q = new_q.copy()
            
            j +=1
            
        if (q_old == q).all(): rj = 1
        else: rj = 0
        return q, rj
    



    def check(self,q ,p, n, picked_q, E_new):
        
        n2 = np.exp(-E_new + self.Hamiltonian.E_old)
        log_sum_exp = -self.Hamiltonian.E_old + np.log(  n + n2 )
        
        if np.exp(-E_new - log_sum_exp) > np.random.uniform():
            
            picked_q = q.copy()
            n+=n2
        
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
        
        q, p = self.Hamiltonian.integrator(q, p, dt, self.model)
        E_new  = self.Hamiltonian.Energy(q, p, self.model)
        stop = self.Hamiltonian.inversion(q, q_2, p, p_2) * int( ( E_new - self.Hamiltonian.E_old)/self.Hamiltonian.E_old  <= self.err)
        
        
            
        if stop:
            q, p = self.model.reflection(q, p)
                
            picked_q, n = self.check(q, p, n, picked_q, E_new)
        
        
        return q, p, n, picked_q, stop 
  
    
  
    
    def inversion(self, q1,q2,p):
        
          return int(( q1-q2) @p > 0)


     

      
        
      
'''   
    def burn_in(self, q):
        stop = True
        dt = self.dt
        
        i = 0
        mean_e = 0
        M, inv_M, log_det_M, dM = self.Hamiltonian.metric(q)
        while stop:
            print("\n", i)
            p = np.ones(self.model.dim)
            
            rn = np.round(np.random.uniform(-1,1, self.model.dim) )*100
            
            p = rn + np.random.normal(0, 1, len(rn))
            
            mass = np.random.uniform(0.1, 1, self.model.dim)
            self.Hamiltonian = C_H(self.model, mass = mass)
            
            E_old = self.Hamiltonian.Energy(q, p, inv_M, log_det_M, dM)
            q2 = q.copy()
            for k in range(1):
                q2, p, _, _, _ = self.Hamiltonian.integrator(q2, p, inv_M, log_det_M, dM, dt)
            E_new  = self.Hamiltonian.Energy(q, p, inv_M, log_det_M, dM)
            print(q2[1::5])
            if check_bounds(q2, self.model.bounds):
                
                q = q2.copy()
                
                err = np.abs((E_new-E_old)/E_old)
                E_old = E_new
                
                mean_e += err
                print(mean_e)
                if i % 100 == 0: 
                    mean_e /=100
                    
                    if mean_e > 0.2:
                        dt /= 10
                    if mean_e < 0.2 and mean_e > 0.05:
                        dt /= 2 
                    if mean_e < 0.05:
                        
                        stop = False
                mean_e = 0       
                i+=1
            self.Hamiltonian =C_H(self.model, mass = 1000)
        return q, dt
'''       
def check_bounds(array, tuple_list):
    return all(bounds[0] <= element <= bounds[1] for element, bounds in zip(array, tuple_list))
            
            
            