import numpy as np
from SignalBHs.GenerateData import generate_data as gen_bh
from SignalBHs.GenerateData import bursts as bur_bh
from SignalCompactBinaries.GenerateData import generate_data  as gen_cb
from SignalCompactBinaries.GenerateData import bursts as bur_cb
  
    
def generate_data(bounds_cb, bounds_bh, Nsources_cb, Nsources_bh, sampling_frequency=16, sigma_noise=1, rng=np.random):

    t, signal_bh, theta_bh, _ = gen_bh(bounds_bh, Nsources_bh, sampling_frequency, sigma_noise, rng)
    
    t, signal_cb, theta_cb, _ = gen_cb(bounds_cb, bounds_bh[1], Nsources_cb, sampling_frequency, sigma_noise, rng)
    
    npoints = int(sampling_frequency*(bounds_cb[1][1] - bounds_cb[1][0]))
    noise = rng.normal(0., sigma_noise, npoints)
    return t, signal_bh + signal_cb, np.concatenate((theta_cb, theta_bh)), noise

def bursts(t, Nsources_cb, Nsources_bh, q):
    return bur_cb(t, Nsources_cb, q[:Nsources_cb*3]) + bur_bh(t, Nsources_bh, q[Nsources_cb*3:])