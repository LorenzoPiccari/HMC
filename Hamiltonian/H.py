import numpy as np

class H():
    
    def __init__(self, M):   
        self.M = M 
        self.inv_M = np.linalg.inv(M)
        
    def momentum_update(self, q, model, rng):
            p = rng.multivariate_normal(np.zeros(len(q)), self.M)
            self.E_old = self.Energy(q, p, model)
            return p
        
    def Energy(self, q, p, model):
            return model.distribution(q) + .5*(p.T @ self.inv_M @ p) 

    
    def integrator(self, q, p, dt, model):
        q, p = Verlet(q, p, dt, self.inv_M, model)
        return q, p
    
    def inversion(self,q1,q2, p1,p2):
        return int( p1 @ (-p2) < 0)
    
    def inversion2(self,q1,q2, p1,p2):
        return int((q1 @ self.inv_M )@ self.int_p < 0)*int( (q2 @ self.inv_M) @ self.int_p < 0)
    
    def inversion3(self,q1,q2, p1, p2, ):
        
        return int( p1 @ (q1-q2) > 0) * int( p2 @ (q2-q1) < 0)
    

    
def Verlet(q, p, dt, inv_M, model):
        p -= (0.5* dt) * model.gradient(q)
        q += dt * (inv_M @ p)
        p -= (0.5*dt) * model.gradient(q)
        return q, p








