# Simplekso algoritmas
import numpy as np


def simplekso_fun(f, x0, delta, eps, Nmax, gamma=2.0, beta=0.5, nu=-0.5, box_limit=10.0):
    x0 = np.array(x0, dtype=float)
    n = len(x0)

    def saugi_f(v):
        if np.any(~np.isfinite(v)):
            return np.inf
        if np.linalg.norm(v, ord=np.inf) > box_limit:
            return np.inf
        reiksme = float(f(v))
        if not np.isfinite(reiksme):
            return np.inf
        return reiksme

    # Pradinis simpleksas: x0 ir po viena taska kiekviena koordinates kryptimi
    simpleksas = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += delta
        simpleksas.append(xi)

    f_reiksmes = [saugi_f(v) for v in simpleksas]
    func_count = len(f_reiksmes)
    trajektorija = [simpleksas[int(np.argmin(f_reiksmes))].copy()]

    k = 0
    while k < Nmax:
        # Surikiuojame virsunes: geriausia pirma, blogiausia pabaigoje
        idx = np.argsort(f_reiksmes)
        simpleksas = [simpleksas[i] for i in idx]
        f_reiksmes = [f_reiksmes[i] for i in idx]

        geriausias = simpleksas[0]
        blogiausias = simpleksas[-1]

        # Sustojimo salygos
        dmax = max(np.linalg.norm(v - geriausias) for v in simpleksas[1:])
        f_sklaida = np.std(f_reiksmes)
        if dmax < eps and f_sklaida < eps:
            break

        # Atspindys per likusiu virsuniu svorio centra (alpha = 1)
        centras = np.mean(simpleksas[:-1], axis=0)
        xr = centras + (centras - blogiausias)
        fr = saugi_f(xr)
        func_count += 1

        if fr < f_reiksmes[0]:
            # Jei atspindys labai geras - ispletimas su gamma
            xe = centras + gamma * (xr - centras)
            fe = saugi_f(xe)
            func_count += 1

            if fe < fr:
                simpleksas[-1] = xe
                f_reiksmes[-1] = fe
            else:
                simpleksas[-1] = xr
                f_reiksmes[-1] = fr

        elif fr < f_reiksmes[-2]:
            # Atspindys geresnis uz bent jau antra blogiausia
            simpleksas[-1] = xr
            f_reiksmes[-1] = fr

        elif fr < f_reiksmes[-1]:
            # Isorinis suspaudimas su beta (0 < beta < 1)
            xc = centras + beta * (xr - centras)
            fc = saugi_f(xc)
            func_count += 1

            if fc <= fr:
                simpleksas[-1] = xc
                f_reiksmes[-1] = fc
            else:
                # Jei suspaudimas nepavyko - maziname simpleksa
                for i in range(1, n + 1):
                    simpleksas[i] = geriausias + 0.5 * (simpleksas[i] - geriausias)
                    f_reiksmes[i] = saugi_f(simpleksas[i])
                func_count += n

        else:
            # Vidinis suspaudimas su beta (0 < beta < 1) - tarp centro ir blogiausio
            xc = centras + beta * (blogiausias - centras)
            fc = saugi_f(xc)
            func_count += 1

            if fc < f_reiksmes[-1]:
                simpleksas[-1] = xc
                f_reiksmes[-1] = fc
            else:
                # Jei suspaudimas nepavyko - maziname simpleksa
                for i in range(1, n + 1):
                    simpleksas[i] = geriausias + 0.5 * (simpleksas[i] - geriausias)
                    f_reiksmes[i] = saugi_f(simpleksas[i])
                func_count += n

        # Atnaujiname taška trajektorijai
        idx = np.argsort(f_reiksmes)
        simpleksas = [simpleksas[i] for i in idx]
        f_reiksmes = [f_reiksmes[i] for i in idx]
        trajektorija.append(simpleksas[0].copy())
        k += 1

        if func_count > 20 * Nmax:
            break

    x_min = simpleksas[0]
    f_min = f_reiksmes[0]
    return x_min, k, f_min, func_count, trajektorija
