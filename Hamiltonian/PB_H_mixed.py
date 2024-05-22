


import numpy as np
from scipy.stats import special_ortho_group

class PB_H_mixed():
    def __init__(self, jump):
        
        self.jump = jump
        self.metric_der = {}
    
    def momentum_update(self, q, model): 
        U = -model.distribution(q)
        self.B = np.random.uniform(1, self.jump, len(q))
        M = self.B/(U + 1)
        
        x = special_ortho_group.rvs(len(q))
        
        #print("\nq",x@np.diag(M)@x.T)
        
        self.R = x
        
        
        p = np.random.multivariate_normal(np.zeros(len(q)), self.R @ np.diag (M) @ self.R.T)
        
        self.int_p = p
        
        self.E_old = U + .5*p.T @ self.R.T @ np.diag (1 / M) @ self.R @ p + .5*np.log(np.prod(M))
        
        return p
        
        
        
    
    def Energy(self, q, p, model):
            return self.E
    
    
    def integrator(self, q, p, dt, model):
        U = -model.distribution(q)
        p = first_step_dq( q, p, dt, model, U, self.B, self.R)
        q, U = second_step_dp( q, p, dt, model, U, self.B, self.R)
        p = third_step_dq( q, p, dt, model, U, self.B, self.R)
        
        M = self.B/(U + 1)
        
        self.E = U + .5*p.T @ self.R.T @ np.diag (1 / M) @ self.R @ p + .5*np.log(np.prod(M))
        self.int_p += p
        return q, p
    
    def inversion(self, q1,q2, p1, p2):
          return int(q1 @ self.int_p < 0)*int( q2 @ self.int_p < 0)
    
def dQVDET(q, model, U, B, R):
    
    grad = np.zeros(len(q))
    grad_logL = - model.gradient(q)
    
    inv_M = R.T @ np.diag((U+1)/B) @ R
    
    for i in range(len(q)):
        grad[i] = np.sum( np.diag(                 inv_M @ R @ np.diag(grad_logL[0]*B/((U+1)**2)) @ R.T               )                    )
    return grad + grad_logL, grad_logL



def dQT(q, p, model, grad_logL, U, B, R):
    
    grad = np.zeros(len(q))
    inv_M = R.T @ np.diag((U+1)/B) @ R
    
    
    for i in range(len(q)):
        grad[i] = -.5*p.T @ inv_M @ R @ np.diag(grad_logL[0]*B/((U+1)**2)) @ R.T  @ inv_M @ p 
    return grad 



def dPT(p, U, B, R):
    return R.T @ np.diag((U+1)/B) @ R @ p 
     



def first_step_dq(q, p, dt, model,  U, B, R):
    derVDET, grad_logL = dQVDET(q, model, U, B, R)
    
    p -= .5 * dt * derVDET
    
    p_1 = p.copy()
    
    stop = 1
    while stop > 0.001:
        s = p.copy()
        p = p_1 - .5 * dt * dQT(q, p, model, grad_logL, U, B, R)
        stop = np.max(np.abs(p-s))
    return p


def second_step_dp(q, p, dt, model,  U, B, R):
    q_1 = q.copy()
    U_1 = U
    
    stop = 1
    while stop > 0.001:
        s = q.copy()
        q = q_1 + .5 * dt * ( dPT(p, U, B , R) + dPT(p, U_1, B, R))
        U = -model.distribution(q)
        stop = np.max(np.abs(q-s))
        
    return q, U


def third_step_dq(q, p, dt, model,  U, B, R):
    derVDET, grad_logL = dQVDET(q, model, U, B, R)
    
    p -= .5 * dt * dQT(q, p, model, grad_logL, U, B, R)
    
    p -= .5 * dt * derVDET

    
        
    return p







