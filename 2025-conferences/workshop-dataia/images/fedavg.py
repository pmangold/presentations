import numpy as np
from tqdm import tqdm

# function to compute gradients
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(X_, y_, theta, lambda_):
    return X_.T @ (sigmoid(X_ @ theta) - y_) / len(y_) + lambda_ * theta    

def logistic_loss(X, y, theta, lambda_):
    n = len(y)
    loss = np.mean(np.log(1 + np.exp(-y * (X @ theta))))  
    reg = (lambda_ / 2) * np.linalg.norm(theta)**2 
    return loss + reg

# function to compute theta_star
def compute_theta_star(data,lambda_,alpha=0.01,tol=1e-12):
    update = tol + 1
    theta = np.zeros(data[0][0].shape[1])
    last_update = np.inf
    n=0
    while update > tol:
        n=n+1
        grad = 0
        for (X, y) in data:
            grad += 1.0/len(data) * gradient(X, y, theta, lambda_)
        theta -= alpha * grad
        update = np.linalg.norm(grad)
        print(update)

        if last_update > update:
            alpha *= 1.01
        else:
            alpha /= 2

        last_update = update

    print("Theta star found in", str(n), "iterations.")
    return theta



# FedAvg
def fed_avg(clients_data,
            theta, step, lambda_,
            n_local, n_rounds,
            seed=None):
    
    rng = np.random.default_rng(seed)

    theta = theta.copy()
    loc_theta = np.array([theta.copy() for _ in range(len(clients_data))])

    hist = [theta.copy()]

    for t in range(n_rounds):

        for c in range(len(clients_data)):
            loc_theta[c] = theta.copy()

            for h in range(n_local):
                grad = gradient(clients_data[c][0], clients_data[c][1],
                                loc_theta[c], lambda_)
                loc_theta[c] = loc_theta[c] - step * grad


        theta = np.mean(loc_theta, axis=0)
        hist.append(theta.copy())

    return np.array(hist)