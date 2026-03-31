import sympy as sp
import grad_nusileidimas

# Kintamieji (skirtingų sienų plotų sumos)
x, y, z = sp.symbols('x y z', real=True)

# 2.1 punktas: paviršiaus ploto sąlyga
surface_constraint = sp.Eq(x + y + z, 1)

# 2.2 punktas: tūrio kvadrato Python funkcija
def V2(x_val, y_val, z_val):
    return x_val * y_val * z_val / 8

# Simbolinė išraiška gradientui ir pertvarkymams
V2_sym = x * y * z / 8

# 3 punktas
z_expr = 1 - x - y
V2_xy_expr = sp.simplify(V2_sym.subs(z, z_expr))

# 4 punktas
f_expr = sp.simplify(-V2_xy_expr)

# 5 punktas
fx_expr = sp.simplify(sp.diff(f_expr, x))
fy_expr = sp.simplify(sp.diff(f_expr, y))

# Paprastos Python funkcijos algoritmui
def z_from_xy(x_val, y_val):
    return 1 - x_val - y_val

def f(X):
    x_val, y_val = X
    return -V2(x_val, y_val, z_from_xy(x_val, y_val))

def gradF(X):
    x_val, y_val = X
    gx = fx_expr.subs({x: x_val, y: y_val})
    gy = fy_expr.subs({x: x_val, y: y_val})
    return [float(gx), float(gy)]

# 6 punktas: reikšmės taškuose
points = {
    "X0": (0, 0),
    "X1": (1, 1),
    "Xm": (0.3, 0.4)   # a = 3, b = 4
}