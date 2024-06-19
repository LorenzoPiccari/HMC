
from Model.Model import Model

import numpy as np
import matplotlib.pyplot as plt


class Linear(Model):
    
    def __init__(self, dim, sample = None, parameters = [1,1] , log_prior = None):
        
        if sample is None: sample = [np.zeros(dim-2),np.zeros(dim-2)]
        bounds = [] 
        np.concatenate([sample[0], np.array([20.0, 90.0])])
        for i in range(dim-2):
            bounds.append((sample[0][i]-10,sample[0][i]+10))
            
        bounds.append((10,60))
        bounds.append((10,60))
        super().__init__( dim, bounds)
        self.npoints=dim
        
        self.err_x = parameters[0]
        self.err_y = parameters[1]
        self.sample = sample
        self.names = ['x' + str(i) for i in range(len(self.sample[0]))]
        self.names = self.names + ['coef', 'intercetta']
        
    
    def name(self):
        return "linear"
    
    def distribution(self, q):
        
        x = (self.sample[0] - q[:self.dim - 2])
        y = (self.sample[1] - ( q[-2]*q[:self.dim - 2] + q[-1] ) )
        
        return (0.5*np.sum( (x/self.err_x)**2 + (y/self.err_y)**2 ) )/self.npoints
    
    

    def gradient(self,q):
        der = np.zeros(len(q))
        
        for i in range(self.dim - 2):
            x = (self.sample[0][i] - q[i])
            y = q[-2]*(self.sample[1][i] - q[-2]*q[i] - q[-1])
            der[i] = x/(self.err_x**2) + y/(self.err_y**2)
            
            
        der[-2] = np.sum(q[:self.dim - 2]*(self.sample[1] - ( q[-2]*q[:self.dim - 2] + q[-1] ) )/(self.err_y**2))
        der[-1] = np.sum((self.sample[1] - (q[-2]*q[:self.dim - 2] + q[-1]))/(self.err_y**2))

        return -der/self.npoints
    
    
    
    def line_plot(self,line = None):
        
        if line is not None: plt.plot(line[0],line[1] ,color='blue', label='Exact Line')
        
        plt.style.use('seaborn-whitegrid')
        
        plt.scatter(self.sample[0], self.sample[1], color='red',alpha = 0.8, label='Sample')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Line and Fake Sample')
        
        
        
    #RETTA TEST
    def line_coeff(self, sample, num_points):

        E_interc=np.mean(sample[:,-1])
        E_coef=np.mean(sample[:, -2])
        E_x = [sample[:,i:-2].mean() for i in range(num_points)]

        return E_interc, E_coef, E_x


    def plot_retta(self, sample, coef, interc,  burn_in = 0, real_xy = None):
            
            _, dim = np.shape(sample)

            E_interc, E_coef, E_x = self.line_coeff(sample[burn_in:,:], dim-2)
            '''
            xx = np.linspace(50, 100, 1000)
            yy = xx*coef + interc
            self.line_plot((xx,yy))
            if real_xy is not None:
                plt.scatter(real_xy[0], real_xy[1], marker='x', color='black',alpha = 0.8, label= "Real_points")
                '''
            plt.plot(E_x, np.array(E_x)*E_coef+E_interc, marker='.', color='green',alpha = 0.8, label= "Retta")
            plt.legend()
            plt.grid(True)