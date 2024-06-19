import Analysis as sis
import time
import numpy as np

def compare(algs, iteration, start_q, burn_in = 0, lag = 100):
    all_m = []
    info = []
    for a in algs:
        print("\n", a.alg_name(), "\n")
        t_i = time.time()
        m, rj = a.run(iteration, start_q)
        all_m.append(m[burn_in:, :])
        t_f = time.time() - t_i
        iat = sis.IAT(m[burn_in:, :], lag)
        info.append([a.alg_name() , t_f, rj, iat])
        np.save(a.alg_name()+".npy", m[burn_in:, :])
        np.save("info"+a.alg_name()+".npy", np.array(info))
    return all_m