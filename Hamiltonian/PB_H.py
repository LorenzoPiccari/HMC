
import numpy as np

class PB_H():
    def __init__(self, jump_max, mass = None, fraction = 1):
        if mass is None:
            self.mass = 1.
        else:
            self.mass = mass
        self.jump_max = jump_max
        self.fraction = fraction
        
        
    def jump(self, dim):
        return np.random.uniform(self.mass, self.mass + self.jump_max, dim)
    
    def momentum_update(self, q, model, rng): 
        
        U = model.distribution(q)
        self.B = self.jump(len(q))
        M = self.B/(U/self.fraction + 1)

        S =  np.eye(len(q)) * M
            
        p = rng.multivariate_normal(np.zeros(len(q)), S)
        
        self.int_p = p
        
        self.E_old = U + .5*p.T @ ((1 / M) * p) + .5*np.log(np.prod(M))
        return p
        

    def Energy(self, q, p, model):
        
            return self.E
    
    
    def integrator(self, q, p, dt, model):
        
        U = model.distribution(q)
        p = first_step_dq( q, p, dt, model, self.B, U, self.fraction)
        q, U = second_step_dp( q, p, dt, model, self.B, U, self.fraction)
        p = third_step_dq( q, p, dt, model, self.B, U, self.fraction)
        
        M = self.B/(U/self.fraction + 1)
        self.E = U + .5*p.T @ np.diag( 1 / M ) @ p + .5*np.log(np.prod(M))
        self.int_p += p
        return q, p
    '''
    def inversion(self, q1,q2, p1, p2):
          return int(q1 @ self.int_p < 0)*int( q2 @ self.int_p < 0)
    '''
    def inversion(self,q1,q2, p1,p2):
        return int( p1 @ (-p2) < 0)
    
def dQVDET(q, model, U, B, f):
    
    grad_V = model.gradient(q)
    grad_det = -len(q) * grad_V/(U/f + 1) 
        
    return grad_V + grad_det, grad_V



def dQT(q, p, model, grad_V, U, B, f):
    
    grad_T = -.5 * (grad_V/f) * ((1/(U/f + 1))**2) * (p.T @ (B*p))
    
    return grad_T 



def dPT(p, B, U, f):
    return ((U/f + 1) / B) * p 
     



def first_step_dq(q, p, dt, model, B, U, f):
    derVDET, grad_V = dQVDET(q, model, U, B, f)
    p -= .5 * dt * derVDET
    
    p_1 = p.copy()
    
    stop = 1
    while stop > 0.001:
        s = p.copy()
        p = p_1 - .5 * dt * dQT(q, p, model, grad_V, U, B, f)
        stop = np.max(np.abs(p-s))
    return p


def second_step_dp(q, p, dt, model,B, U, f):
    q_1 = q.copy()
    U_1 = U
    
    stop = 1
    while stop > 0.001:
        s = q.copy()
        q = q_1 + .5 * dt * ( dPT(p, B, U, f) + dPT(p,B, U_1, f))
        U = model.distribution(q)
        stop = np.max(np.abs(q-s))
        
    return q, U


def third_step_dq(q, p, dt, model, B, U, f):
    derVDET, grad_V = dQVDET(q, model, U, B, f)
    
    p -= .5 * dt * dQT(q, p, model, grad_V,U, B, f)
    
    p -= .5 * dt * derVDET

    
        
    return p












