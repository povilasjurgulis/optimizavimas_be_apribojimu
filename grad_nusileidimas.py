# Gradientinio nusileidimo algoritmas
import numpy as np

# gamma - žingsnio daugiklis; 
def grad_fun(f, gradF, x0, gamma, eps, Nmax):
    x = x0
    k = 0

    while k < Nmax:
        g = gradF(x)
        if np.linalg.norm(g) < eps: # jeigu gradiento norma mažesnė nei epsilon
            break
        x = x - gamma * g

    return x