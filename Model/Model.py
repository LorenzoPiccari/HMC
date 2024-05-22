
import numpy as np


    
class Model():
    
    def __init__(self, dim, bounds = None):
        
        self.dim = dim
        
        self.bounds = self.create_bounds(bounds)
        
        self.names = ['x' + str(i) for i in range(self.dim)]
    

    
    def create_bounds(self, bounds):
        if bounds is None: return [(-2,2) for i in range(self.dim)]
        else: 
            if isinstance(bounds[0], tuple):
                return bounds
            else: return [(bounds[i]-5 , bounds[i]+5 ) for i in range(self.dim)]
        
        
    def in_bounds(self, q):
        return all(self.bounds[i][0] < q[i] < self.bounds[i][1] for i in range(len(q)))


    def reflection(self, q, p):
        return q, p
        
    #THOSE 3 DEF ARE NECESSARY FOR HMC
    def distribution(self, q): #-logL
        pass
    
    def gradient(self, q): #grad-logL
        return approx_gradient(q, self.distribution)
    
    
def approx_gradient(q, funct):
    return np.array([approx_grad_i(i, q, funct)[1] for i in range(len(q))])
    
def approx_grad_i(i, q, funct):
    step = q*0
    step[i] = 0.01
    return np.gradient(np.array([funct(q-step), funct(q), funct(q+step)], dtype = float), np.array( [q[i] - step[i], q[i] , q[i] + step[i]], dtype = float) )
   