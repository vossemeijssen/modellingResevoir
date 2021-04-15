import numpy as np
import matplotlib.pyplot as plt
import tqdm
from reservoirModule import *

# Variables
L = 10  # Total length
dx = 0.005  # distance step
t_tot = 0.02  # Total time

# Moeten worden gefinetuned:
c = Constants(
    phi = 0.1,  # Porosity
    u_inj = 1.0,  # Water injection speed
    mu_w = 1.e-3,
    mu_o = 0.04,
    kappa = 1,
    k_rw0 = 1,
    k_ro0 = 1,
    n_w = 4,  # >=1
    n_o = 2,  # >=1
    S_or = 0.1,  # Oil rest-saturation
    S_wc = 0.1,
    sigma = 0.1,
    labda=2)

def magic_function(x, c):
    return df_dSw(x, c) - (f_w(x, c) - f_w(c.S_wc, c))/(x - c.S_wc)

S_w_shock = bisection(magic_function, (c.S_wc, 1 - c.S_or), 100, c)
shockspeed = c.u_inj/c.phi*df_dSw(S_w_shock, c)
dt = dx/shockspeed  # time step

# Code
N = int(L/dx)
time_N = int(t_tot / dt)
S_w = np.ones(N) * c.S_wc
S_w[0] = 1 - c.S_or
S_w_all = [S_w]

print("Shockspeed =", shockspeed)
print("L =", L)
print("dx=", dx)
print("N =", N)
print("T =", t_tot)
print("dt=", dt)
print("tN=", time_N)

for t in tqdm.tqdm(range(time_N)):
    newS_w = np.copy(S_w)
    for i in range(1, N-1):
        dSw_dx = (-S_w[i-1] + S_w[i+1]) / (2 * dx)
        dS_w = 0.1

        # implementation of Lax–Friedrichs Method
        newS_w[i] = (S_w[i-1]+S_w[i+1])/2 - dt/2/dx *c.u_inj/c.phi *(f_w(S_w[i+1], c)-f_w(S_w[i-1], c)) - D_cap(S_w[i], c)*dt/dx*(S_w[i-1]-2*S_w[i]+S_w[i+1])
        # newS_w[i] = (S_w[i-1]+S_w[i])/2 - dt/2/dx*u_inj/phi *(f_w(S_w[i])-f_w(S_w[i-1]))

    S_w = newS_w
    S_w_all.append(newS_w)

# S_w_all = np.matrix(S_w_all)
# plt.matshow(S_w_all)
# #plt.contour(S_w_all, np.linspace(0.8, 0.9, 100))
# plt.colorbar()
# plt.show()
plt.figure()
print(len(S_w))
plt.plot(np.linspace(0, L, N), S_w)

# Analytic solution:
# outer part:
# phi*dS_w/dt + du_w/dx = 0, S_w(0, t) = 1, S_w(x, 0) = S_wc
# Gives dS_w/deta = 0 or eta = uinj/phi * df_w/dS_w
x = np.linspace(0.9, 0.1 ,1000, endpoint=False)
y = [c.u_inj/c.phi*df_dSw(xi, c)*t_tot for xi in x]

analytical_solution_x = []
analytical_solution_y = []
shockpoint = shockspeed*t_tot
before_shock = True
i = 0
while before_shock:
    if y[i] < shockpoint:
        analytical_solution_x.append(y[i])
        analytical_solution_y.append(x[i])
    else:
        analytical_solution_x.append(shockpoint)
        analytical_solution_y.append(S_w_shock)
        analytical_solution_x.append(shockpoint)
        analytical_solution_y.append(c.S_wc)
        analytical_solution_x.append(L)
        analytical_solution_y.append(c.S_wc)
        before_shock = False
    i += 1

plt.plot(analytical_solution_x, analytical_solution_y)
plt.legend(['Numerical approximation', 'Analytical solution'])
plt.title("Saturation of water")
plt.xlabel('Length (meter)')
plt.ylabel("Water saturation")
plt.show()