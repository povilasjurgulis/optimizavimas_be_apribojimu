# Auksinio pjūvio algoritmas

import math # dėl šaknies

def auks_pjuv_fun(f, l, r):
    t = (math.sqrt(5) - 1)/2 # "tau"
    e = 0.0001
    L = r - l
    x1 = r - t*L
    x2 = l + t*L
    f1 = f(x1)
    f2 = f(x2)
    func_count = 2
    it = 0
    save_x = []
    save_x.append(x1)
    save_x.append(x2)
    while(L >= e):
        if f2 < f1:
            l = x1
            L = r - l
            x1 = x2
            f1 = f2
            x2 = l + t*L
            f2 = f(x2)
            save_x.append(x2)
        else:
            r = x2
            L = r - l
            x2 = x1
            f2 = f1
            x1 = r - t*L
            f1 = f(x1)
            save_x.append(x1)
        it += 1
        func_count += 1
    if f1 < f2:
        return x1, it, func_count, save_x
    else:
        return x2, it, func_count, save_x