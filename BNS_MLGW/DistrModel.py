import numpy as np
import mlgw
from Model.Model import Model

class Posterior(Model):

    def __init__(self, variance=None):

    def GenWaveform(self, m1, m2, s1, s2):
        
        


    def loglikelihood

    def logprior(self, :

    def distribution(self):
        return loglikelihood + logprior
    
    def gradient(self,q):
        return (self.inv_cov @ (q-self.mean))
