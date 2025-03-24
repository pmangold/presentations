import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# FedAvg
def fed_avg(gradients,
            theta, step, lambda_, 
            n_local, n_rounds,
            sigma=0,
            seed=None,
            theta_star=None):

    rng = np.random.default_rng(seed)

    if theta_star is None:
        theta_star = np.zeros(np.array(theta).shape)

    theta = np.array(theta, dtype=float)
    theta = theta.copy()
    loc_theta = np.array([theta.copy() for _ in range(len(gradients))])

    hist = [theta.copy()]
    hist_loc = [loc_theta.copy()]

    lyap = [np.linalg.norm(theta - theta_star) ** 2]

    for t in range(n_rounds):


        for c, grad in enumerate(gradients):

            loc_theta[c] = theta.copy()

            for h in range(n_local):
                loc_theta[c] = loc_theta[c] - step * (grad(loc_theta[c]) + np.random.normal(scale=sigma, size=len(theta)))

        theta = np.mean(loc_theta, axis=0)

        hist.append(theta.copy())
        hist_loc.append(loc_theta.copy())
        lyap.append(np.linalg.norm(theta-theta_star) ** 2)

    return np.array(hist), np.array(lyap), np.array(hist_loc)


# Scaffold
def scaffold(gradients,
             theta, step, lambda_,
             n_local, n_rounds,
             sigma=0,
             seed=None,
             theta_star=0):

    def lyap_func(theta, xi):
        return np.linalg.norm(theta-theta_star)**2 + step**2*n_local**2 * np.linalg.norm(xi-xi_star)**2/N

    if theta_star is None:
        theta_star = np.zeros(np.array(theta).shape)    
    
    rng = np.random.default_rng(seed)
    N = len(gradients)
    
    theta = np.array(theta, dtype=float)
    theta = theta.copy()
    loc_theta = np.array([theta.copy() for _ in range(len(gradients))])
    xi = np.array([np.zeros(theta.shape) for _ in range(len(gradients))])
    xi_star = np.array([grad(theta_star) for grad in gradients])

    
    hist = [theta.copy()]
    hist_loc = [loc_theta.copy()]
    lyap = [lyap_func(theta, xi)]
            
    for t in range(n_rounds):

        for c, grad in enumerate(gradients):

            loc_theta[c] = theta.copy()

            for h in range(n_local):
                loc_theta[c] = loc_theta[c] - step * ( grad(loc_theta[c])  - xi[c] + np.random.normal(scale=sigma, size=len(theta)))

        theta = np.mean(loc_theta, axis=0)
        xi += 1.0 / (step * n_local) * (theta - loc_theta)

        hist.append(theta.copy())
        hist_loc.append(loc_theta.copy())
        lyap.append(lyap_func(theta, xi))
        


    return np.array(hist), np.array(lyap), np.array(hist_loc)

