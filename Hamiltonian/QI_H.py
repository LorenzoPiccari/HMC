import numpy as np

    
class QI_H():
    
    def __init__(self):
        pass
    
    def momentum_update(self, q, model):
                
            M, self.inv_M = self.metric(len(q))
        
            p = np.random.multivariate_normal(np.zeros(len(q)), np.eye(len(q))*M)
            
            self.E_old = self.Energy(q, p, model)
            return p
        
    
    def Energy(self, q, p, model):
            return - model.distribution(q) + (.5*p *self.inv_M) @ p
        
    
    def integrator(self, q, p, dt, model):
        return Verlet(q, p, dt, self.inv_M, model)
    
    def metric(self, dim):
        
        M = np.random.uniform(0.1, 100, dim)
        
        inv_M = np.eye(dim)/M
        
        return M, inv_M
    


def gradient_p(p, inv_M):
    return inv_M @ p



def Euler(q, p, dt, inv_M, model):
        p -= dt * (-model.gradient(q))
        q += dt * gradient_p(p, inv_M)
        
        return q, p
    
    
def Verlet(q, p, dt, inv_M, model):
        p -= (0.5* dt) * model.gradient(q)
        q += dt * gradient_p(p, inv_M)
        p -= (0.5*dt) * model.gradient(q)
        return q, p


