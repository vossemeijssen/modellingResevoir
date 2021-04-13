import numpy as np
import matplotlib.pyplot as plt


# Variables
u_inj = 1.0  # Water injection speed
S_or = 0.1  # Oil rest-saturation
S_wc = 0.1  # Water capillary saturation
L = 10  # Total length
dx = 0.1  # distance step
t_tot = 1  # Total time
dt = 0.01  # time step
phi = 0.2  # Porosity
# Moeten worden gefinetuned:
mu_w = 1
mu_o = 1
kappa = 1
k_rw0 = 1
k_ro0 = 1
n_w = 1  # >=1
n_o = 1  # >=1



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
    # df_dSwn = C_w * C_o / (C_o + C_w * (S_wn)**n_w / ((1 - S_wn) ** n_o)) * (n_o / (1-S_wn) + n_w /S_wn)
    df_dSwn = C_w * C_o / (C_o * (S_wn - 1) - C_w * S_wn)**2
    return df_dSwn / (1 - S_or - S_wc)

# Code
N = int(L/dx)
S_w = np.ones(N) * S_wc
S_w[0] = 1 - S_or
S_w_all = [S_w]

t = 0
while t < t_tot:
    newS_w = np.copy(S_w)
    for i in range(1, N-1):
        dSw_dx = (-S_w[i-1] + S_w[i+1]) / (2 * dx)
        dS_w = 0.1
        df_dSw2 = (-f_w(S_w[i] - dS_w) + f_w(S_w[i] + dS_w)) / (2 * dS_w)
        # newS_w[i] = S_w[i] - dt * u_inj * df_dSw(S_w[i]) * dSw_dx
        #newS_w[i] = S_w[i] - dt * u_inj * df_dSw2 * dSw_dx

        # implementation of Lax–Friedrichs Method
        newS_w[i] = (S_w[i-1]+S_w[i+1])/2 - dt/2/dx*u_inj/phi *(f_w(S_w[i+1])-f_w(S_w[i-1]))
    S_w = newS_w
    S_w_all.append(newS_w)
    t += dt

S_w_all = np.matrix(S_w_all)
plt.matshow(S_w_all)
plt.colorbar()
plt.show()

plt.plot(S_w)
plt.show()