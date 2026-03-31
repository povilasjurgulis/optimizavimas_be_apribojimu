# Gradientinio nusileidimo algoritmas
import numpy as np

# gamma - žingsnio daugiklis; 
def grad_fun(f, gradF, x0, gamma, eps, Nmax):
    x = np.array(x0, dtype=float)
    k = 0
    func_count = 0
    grad_count = 0
    trajektorija = [x.copy()]
    f_curr = float(f(x))
    func_count += 1

    while k < Nmax:
        g = np.array(gradF(x), dtype=float)
        grad_count += 1
        if np.any(~np.isfinite(g)) or np.any(~np.isfinite(x)):
            break
        if np.linalg.norm(g) < eps: # jeigu gradiento norma mažesnė nei epsilon
            break

        kandidatas = x - gamma * g
        fk = float(f(kandidatas))
        func_count += 1
        if not np.isfinite(fk):
            break
        
        # Jeigu funkcija didėja, stabdome (divergencija)
        if fk > f_curr:
            break
            
        x = kandidatas
        f_curr = fk
        k = k + 1
        trajektorija.append(x.copy())

    f_min = float(f(x))
    func_count += 1

    return x, k, f_min, func_count, grad_count, trajektorija