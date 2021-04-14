import numpy as np
import matplotlib.pyplot as plt
import tqdm


# Variables
u_inj = 1.0  # Water injection speed
S_or = 0.1  # Oil rest-saturation
S_wc = 0.1  # Water capillary saturation
L = 5  # Total length
dx = 0.01  # distance step
t_tot = 1  # Total time
phi = 0.1  # Porosity

# Moeten worden gefinetuned:
mu_w = 1.e-3
mu_o = 0.4
kappa = 1
k_rw0 = 1
k_ro0 = 1
n_w = 4  # >=1
n_o = 2  # >=1


# Functions
def D_cap(S_w):
    return 0


def f_w(S_w):
    S_wn = (S_w - S_wc)/(1 - S_or - S_wc)
    l_w = kappa / mu_w * k_rw0 * (S_wn) ** n_w
    l_o = kappa / mu_o * k_ro0 * (1-S_wn) ** n_o
    return l_w / (l_w + l_o)


def df_dSw(S_w):
    C_o = k_ro0 / mu_o
    C_w = k_rw0 / mu_w
    S_wn = (S_w - S_wc) / (1 - S_or - S_wc)
    df_dSwn = C_w * C_o / (C_o + C_w * (S_wn)**n_w / ((1 - S_wn) ** n_o)) * (n_o / (1-S_wn) + n_w /S_wn)
    # df_dSwn = C_w * C_o / (C_o * (S_wn - 1) - C_w * S_wn)**2
    return df_dSwn / (1 - S_or - S_wc)


def bisection(f, startpoints, n):
    x1 = startpoints[0]
    x2 = startpoints[1]
    for i in range(n):
        x3 = (x1 + x2)/ 2
        if f(x3) > 0:
            x1 = x3
        else:
            x2 = x3
    return (x1 + x2) / 2


def magic_function(x):
    return df_dSw(x) - (f_w(x) - f_w(S_wc))/(x - S_wc)


S_w_shock = bisection(magic_function, (S_wc, 1 - S_or), 100)
shockspeed = df_dSw(S_w_shock)
dt = dx/shockspeed  # time step

# Code
N = int(L/dx)
time_N = int(t_tot / dt)
S_w = np.ones(N) * S_wc
S_w[0] = 1 - S_or
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
        # implementation using direct calculation of df/dSw
        # newS_w[i] = S_w[i] - dt * u_inj * df_dSw(S_w[i]) * dSw_dx

        # implementation using numerical approximation of df/dSw
        # df_dSw2 = (-f_w(S_w[i] - dS_w) + f_w(S_w[i] + dS_w)) / (2 * dS_w)
        # newS_w[i] = S_w[i] - dt * u_inj * df_dSw2 * dSw_dx

        # implementation of Lax–Friedrichs Method
        newS_w[i] = (S_w[i-1]+S_w[i+1])/2 - dt/2/dx *(f_w(S_w[i+1])-f_w(S_w[i-1]))
        # newS_w[i] = (S_w[i-1]+S_w[i])/2 - dt/2/dx*u_inj/phi *(f_w(S_w[i])-f_w(S_w[i-1]))

    S_w = newS_w
    S_w_all.append(newS_w)

S_w_all = np.matrix(S_w_all)
# plt.matshow(S_w_all)
plt.contour(S_w_all, np.linspace(0.8, 0.9, 100))
plt.colorbar()
plt.show()
plt.figure()
plt.plot(np.linspace(0, L, N), S_w)
plt.show()