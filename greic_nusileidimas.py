# Greičiausio nusileidimo algoritmas
import numpy as np
import auksinis_pjuvis

def greic_nus(f, gradF, x0, eps, Nmax):
    x = np.array(x0, dtype=float)
    k = 0
    func_count = 0
    grad_count = 0
    trajektorija = [x.copy()]

    while k < Nmax:
        g = np.array(gradF(x), dtype=float)
        grad_count += 1
        if np.linalg.norm(g) < eps:
            break

        kryptis = -g

        def phi(ilgis):
            return float(f(x + ilgis * kryptis))

        ilgis, _, papildomi_f, _ = auksinis_pjuvis.auks_pjuv_fun(phi, 0.0, 1.0)
        func_count += papildomi_f

        x = x + ilgis * kryptis
        k += 1
        trajektorija.append(x.copy())

    f_min = float(f(x))
    func_count += 1

    return x, k, f_min, func_count, grad_count, trajektorija