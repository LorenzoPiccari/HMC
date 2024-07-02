import numpy as np
from tqdm import tqdm

class MC_Algorithm():



    def __init__(self, user_model, rng):
            self.rng = rng
            self.model = user_model

    def prep_run(self, start_q, iteration):
        sample = np.zeros((iteration, self.model.dim) )
        
        rejected = 0
        
        if start_q is None:
                x = self.new_point()
                sample[-1] = x
        else:
            sample[-1] = start_q
            
        return sample, rejected
            
    def run(self, iteration, start_q = None):
        
            sample, rejected = self.prep_run(start_q, iteration)
            
            
            for i in tqdm(range(iteration)):
                
                sample[i], rj = self.Kernel(sample[i-1,:])
                
                rejected+= rj
                
            
            return sample, rejected
    
    def run2(self, iteration, start_q = None):
        
            sample, rejected = self.prep_run(start_q, iteration)
            
            
            for i in (range(iteration)):
                
                sample[i], rj = self.Kernel(sample[i-1,:])
                
                rejected+= rj
                
            
            return sample, rejected
    
    def simple_run(self, iteration, start_q):
        
            sample = np.zeros((iteration, self.model.dim) )
            
            rejected = 0
            sample[-1] = start_q
            
            for i in (range(iteration)):
                
                sample[i], rj = self.Kernel(sample[i-1,:])
                
                rejected+= rj
            
            return sample    
    
    def new_point(self):
            generate_random_numbers = lambda pair: self.rng.uniform(pair[0], pair[1])
            x = list(map(generate_random_numbers, self.model.bounds))
            return np.array(x)
    
    def acceptance(self, q, new_q, alpha): #alpha definito come log(denominatore)-log(numeratore)
            
            if alpha < self.rng.exponential(1):
                    return new_q, 0
            else: return q, 1
            
            
        
        
    
    

