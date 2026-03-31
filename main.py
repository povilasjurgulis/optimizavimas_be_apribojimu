import sympy as sp
import numpy as np
import grad_nusileidimas, greic_nusileidimas, simplekso

# Kintamieji (skirtingų sienų plotų sumos)
x, y, z = sp.symbols('x y z', real=True)

# 2.1 punktas: paviršiaus ploto sąlyga
surface_constraint = sp.Eq(x + y + z, 1)

# 2.2 punktas: tūrio kvadrato funkcija
def V2(x_val, y_val, z_val):
    return x_val * y_val * z_val / 8

# Išraiška gradientui ir pertvarkymams
V2_sym = x * y * z / 8

# 3 punktas
z_expr = 1 - x - y
V2_xy_expr = sp.simplify(V2_sym.subs(z, z_expr))

# 4 punktas
f_expr = sp.simplify(-V2_xy_expr)

# 5 punktas
fx_expr = sp.simplify(sp.diff(f_expr, x))
fy_expr = sp.simplify(sp.diff(f_expr, y))

def z_from_xy(x_val, y_val):
    return 1 - x_val - y_val

def f(X):
    x_val, y_val = X
    return float(-V2(x_val, y_val, z_from_xy(x_val, y_val)))

def gradF(X):
    x_val, y_val = X
    gx = fx_expr.subs({x: x_val, y: y_val})
    gy = fy_expr.subs({x: x_val, y: y_val})
    return np.array([float(gx), float(gy)], dtype=float)


# Tikslo funkcija ir jos gradientas (be apribojimų)
f_opt = f
gradF_opt = gradF

# 6 punktas: reikšmės taškuose
points = {
    "X0": (0, 0),
    "X1": (1, 1),
    "Xm": (0.3, 0.4)   # a = 3, b = 4
}


def spausdinti_tasku_reiksmes():
    print("6: Pradzios taskai\n")
    for pavadinimas, taskas in points.items():
        taskas_np = np.array(taskas, dtype=float)
        f_reiksme = f(taskas_np)
        grad_reiksme = gradF(taskas_np)
        print(
            f"   {pavadinimas} = {taskas}: f = {f_reiksme:.8f}, grad = [{grad_reiksme[0]:.8f} {grad_reiksme[1]:.8f}]"
        )
    print()


def paleisti_optimizavima():
    rezultatai = {
        "gradientinis": {},
        "greiciausias": {},
        "simpleksas": {},
    }

    eps = 1e-6
    Nmax = 5000

    for pavadinimas, taskas in points.items():
        x0 = np.array(taskas, dtype=float)

        # Gradientinis nusileidimas
        x_min, it, f_min, f_count, g_count, traj = grad_nusileidimas.grad_fun(
            f=f_opt,
            gradF=gradF_opt,
            x0=x0,
            gamma=0.2,
            eps=eps,
            Nmax=Nmax,
        )
        rezultatai["gradientinis"][pavadinimas] = {
            "x_min": x_min,
            "f_min": f_min,
            "iteracijos": it,
            "f_kiekis": f_count,
            "g_kiekis": g_count,
            "trajektorija": traj,
        }

        # Greičiausias nusileidimas (žingsnis randamas auksinio pjūvio metodu)
        x_min, it, f_min, f_count, g_count, traj = greic_nusileidimas.greic_nus(
            f=f_opt,
            gradF=gradF_opt,
            x0=x0,
            eps=eps,
            Nmax=Nmax,
        )
        rezultatai["greiciausias"][pavadinimas] = {
            "x_min": x_min,
            "f_min": f_min,
            "iteracijos": it,
            "f_kiekis": f_count,
            "g_kiekis": g_count,
            "trajektorija": traj,
        }

        # Deformuojamas simpleksas
        x_min, it, f_min, f_count, traj = simplekso.simplekso_fun(
            f=f_opt,
            x0=x0,
            delta=0.2,
            eps=eps,
            Nmax=Nmax,
        )
        rezultatai["simpleksas"][pavadinimas] = {
            "x_min": x_min,
            "f_min": f_min,
            "iteracijos": it,
            "f_kiekis": f_count,
            "g_kiekis": 0,
            "trajektorija": traj,
        }

    return rezultatai


def spausdinti_rezultatus(rezultatai):
    print("7: Optimizavimo rezultatai\n")
    
    # Pavadinimų žemėlapis
    alg_names = {
        "gradientinis": "Gradientinis nusileidimas",
        "greiciausias": "Greičiausias nusileidimas",
        "simpleksas": "Simpleksas",
    }
    
    for alg_key in ["gradientinis", "greiciausias", "simpleksas"]:
        alg_rez = rezultatai[alg_key]
        print(f"{alg_names[alg_key]}:")
        
        for tasko_pav in ["X0", "X1", "Xm"]:
            if tasko_pav not in alg_rez:
                continue
                
            r = alg_rez[tasko_pav]
            x_min = r["x_min"]
            x_opt = x_min[0]
            y_opt = x_min[1]
            f_val = r["f_min"]
            
            print(
                f"   {tasko_pav} -> Sprendinys: [{x_opt:.8f} {y_opt:.8f}], "
                f"f: {f_val:.8f}, Iteraciju: {r['iteracijos']}, Gradiento iskvietimu: {r['g_kiekis']}"
            )
        print()


def braizyti(rezultatai, rezimas="4"):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("9. Vizualizacija praleista (nepavyko importuoti matplotlib):", exc)
        return

    # Tikslo funkcijos kontūrai (vienoda sritis visiems grafikams)
    x_min, x_max = -0.2, 1.2
    y_min, y_max = -0.2, 1.2
    x_vals = np.linspace(x_min, x_max, 300)
    y_vals = np.linspace(y_min, y_max, 300)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = X * Y * (1 - X - Y) / 8.0

    algoritmai = ["gradientinis", "greiciausias", "simpleksas"]
    alg_names = {
        "gradientinis": "Gradientinis nusileidimas",
        "greiciausias": "Greičiausias nusileidimas",
        "simpleksas": "Simpleksas",
    }
    pradzia_config = {
        "X0": {"marker": "o", "color": "#1f77b4"},
        "X1": {"marker": "s", "color": "#ff7f0e"},
        "Xm": {"marker": "^", "color": "#2ca02c"},
    }
    alg_linijos = {
        "gradientinis": "-",
        "greiciausias": "--",
        "simpleksas": ":",
    }
    alg_spalvos = {
        "gradientinis": "#1f77b4",
        "greiciausias": "#ff7f0e",
        "simpleksas": "#2ca02c",
    }

    def nubraizyti_aši(ax, alg_key):
        alg_rez = rezultatai[alg_key]

        cont = ax.contourf(X, Y, Z, levels=40, cmap="RdYlBu_r", vmin=np.min(Z), vmax=np.max(Z))
        plt.colorbar(cont, ax=ax, label="V^2(x,y)")

        pilnos_trajektorijos = {}
        for tasko_pav in ["X0", "X1", "Xm"]:
            if tasko_pav not in alg_rez:
                continue

            r = alg_rez[tasko_pav]
            traj = np.array(r["trajektorija"], dtype=float)
            pilnos_trajektorijos[tasko_pav] = traj
            valid_mask = (
                np.isfinite(traj).all(axis=1)
                & (traj[:, 0] >= x_min) & (traj[:, 0] <= x_max)
                & (traj[:, 1] >= y_min) & (traj[:, 1] <= y_max)
            )
            traj_valid = traj[valid_mask]

            if len(traj_valid) > 0:
                config = pradzia_config[tasko_pav]
                ax.plot(
                    traj_valid[:, 0],
                    traj_valid[:, 1],
                    marker=config["marker"],
                    markersize=4,
                    linewidth=2.0,
                    linestyle=alg_linijos[alg_key],
                    color=config["color"],
                    label=f"{tasko_pav}",
                    alpha=0.9,
                )

        ax.set_title(f"{alg_names[alg_key]}", fontsize=13, fontweight='bold')
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=10, framealpha=0.95)
        ax.grid(alpha=0.3, linestyle="--")

        inset = ax.inset_axes([0.62, 0.05, 0.35, 0.35])
        inset.set_title("Visi taškai", fontsize=7)
        for tasko_pav, traj in pilnos_trajektorijos.items():
            finite_mask = np.isfinite(traj).all(axis=1)
            traj_full = traj[finite_mask]
            if len(traj_full) == 0:
                continue
            config = pradzia_config[tasko_pav]
            inset.plot(
                traj_full[:, 0],
                traj_full[:, 1],
                marker=config["marker"],
                markersize=2.5,
                linewidth=1.0,
                linestyle=alg_linijos[alg_key],
                color=config["color"],
                alpha=0.9,
            )
        inset.grid(alpha=0.2, linestyle="--")
        inset.tick_params(labelsize=7)

    if rezimas in ["1", "2", "3"]:
        pasirinktas = {
            "1": "gradientinis",
            "2": "greiciausias",
            "3": "simpleksas",
        }[rezimas]
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        nubraizyti_aši(ax, pasirinktas)
        plt.suptitle(f"9: {alg_names[pasirinktas]}", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        return

    if rezimas == "5":
        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        cont = ax.contourf(X, Y, Z, levels=40, cmap="RdYlBu_r", vmin=np.min(Z), vmax=np.max(Z))
        plt.colorbar(cont, ax=ax, label="V^2(x,y)")

        for alg_key in algoritmai:
            alg_rez = rezultatai[alg_key]
            for tasko_pav in ["X0", "X1", "Xm"]:
                if tasko_pav not in alg_rez:
                    continue
                traj = np.array(alg_rez[tasko_pav]["trajektorija"], dtype=float)
                valid_mask = np.isfinite(traj).all(axis=1)
                traj_valid = traj[valid_mask]
                if len(traj_valid) == 0:
                    continue
                cfg = pradzia_config[tasko_pav]
                ax.plot(
                    traj_valid[:, 0],
                    traj_valid[:, 1],
                    marker=cfg["marker"],
                    markersize=3,
                    linewidth=1.6,
                    linestyle=alg_linijos[alg_key],
                    color=alg_spalvos[alg_key],
                    alpha=0.85,
                    label=f"{alg_names[alg_key]} - {tasko_pav}",
                )

        ax.set_title("Visi algoritmai viename grafike", fontsize=13, fontweight='bold')
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3, linestyle="--")

        handles, labels = ax.get_legend_handles_labels()
        unikalus = dict(zip(labels, handles))
        ax.legend(unikalus.values(), unikalus.keys(), loc="best", fontsize=8, framealpha=0.95)
        plt.suptitle("9: Tikslo funkcija ir optimizavimo trajektorijos", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax_idx, alg_key in enumerate(algoritmai):
        ax = axes[ax_idx]
        nubraizyti_aši(ax, alg_key)
    
    plt.suptitle("9: Tikslo funkcija ir optimizavimo trajektorijos", fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    spausdinti_tasku_reiksmes()
    rez = paleisti_optimizavima()
    spausdinti_rezultatus(rez)

    print("\n9 užduotis: pasirinkite vizualizaciją")
    print("1 - Tik Gradientinis nusileidimas")
    print("2 - Tik Greičiausias nusileidimas")
    print("3 - Tik Simpleksas")
    print("4 - Visi algoritmai atskiruose grafikuose")
    print("5 - Visi algoritmai viename grafike")
    print("0 - Nevizualizuoti")

    pasirinkimas = input("Jūsų pasirinkimas [0-5]: ").strip()
    if pasirinkimas == "0":
        print("Vizualizacija praleista.")
    elif pasirinkimas in ["1", "2", "3", "4", "5"]:
        braizyti(rez, pasirinkimas)
    else:
        print("Neteisingas pasirinkimas. Rodomi visi algoritmai atskirai.")
        braizyti(rez, "4")