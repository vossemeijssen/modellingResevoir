import numpy as np
import matplotlib.pyplot as plt
import tqdm
from reservoirModule import *

# Variables
u_inj = 1.0  # Water injection speed
S_or = 0.1  # Oil rest-saturation
S_wc = 0.1  # Water capillary saturation
L = 1  # Total length
dx = 0.005  # distance step
t_tot = 0.02  # Total time
phi = 0.1  # Porosity

# Moeten worden gefinetuned:
mu_w = 1.e-3
mu_o = 0.04
kappa = 1
k_rw0 = 1
k_ro0 = 1
n_w = 4  # >=1
n_o = 2  # >=1

def magic_function(x):
    return df_dSw(x) - (f_w(x) - f_w(S_wc))/(x - S_wc)

S_w_shock = bisection(magic_function, (S_wc, 1 - S_or), 100)
shockspeed = u_inj/phi*df_dSw(S_w_shock)

dx = 0.05
dt = dx/shockspeed/4 # time step


# Code
N = int((L-1)/dx)
r = np.linspace(1,L,N)
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
    newRS_w = np.copy(S_w)
    newS_w  = np.copy(S_w)
    for i in range(1, N-1):
        # implementation of Lax–Friedrichs Method
        newRS_w[i] = (r[i-1]*S_w[i-1]+r[i+1]*S_w[i+1])/2 - dt/2/dx * u_inj/phi *(f_w(S_w[i+1])*r[i+1]-f_w(S_w[i-1])*r[i-1])

        newS_w[i] = newRS_w[i]/r[i];
    S_w = newS_w
    S_w_all.append(newS_w)

print(len(S_w))
plt.matshow(S_w_all)
plt.show()
plt.plot(np.linspace(0, L, N), S_w)
plt.show()