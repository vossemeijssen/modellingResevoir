import numpy as np
import matplotlib.pyplot as plt
import tqdm
from reservoirModule import *

# Variables
L = 10  # Total length
dx = 0.1  # distance step
t_tot = 0.3  # Total time

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
    S_wc = 0.1 )


def magic_function(x, c):
    return df_dSw(x, c) - (f_w(x, c) - f_w(c.S_wc, c))/(x - c.S_wc)

S_w_shock = bisection(magic_function, (c.S_wc, 1 - c.S_or), 100, c)
shockspeed = c.u_inj/c.phi*df_dSw(S_w_shock, c)
dt = dx/shockspeed  # time step
k = 0.9/2

# Code
N = int(L/dx)
time_N = int(t_tot / dt)
S_w = np.ones((N,N)) * c.S_wc
for j in range(N):
    M = 1+int(5*(1+np.sin(j*k)))
    for i in range(M):
        S_w[i,j] = 1 - c.S_or
S_w_all = [S_w]

plt.contourf(S_w)
plt.show()
plt.figure()

print("Shockspeed =", shockspeed)
print("Shock position = ",shockspeed*t_tot)
print("L =", L)
print("dx=", dx)
print("N =", N)
print("T =", t_tot)
print("dt=", dt)
print("tN=", time_N)

for t in tqdm.tqdm(range(time_N)):
    newS_w = np.copy(S_w)
    for j in range(0, N):
        M = 1 + int(5 * (1 + np.sin(j * k)))
        for i in range(M, N-1):
            # implementation of Lax–Friedrichs Method
            newS_w[i,j] = ( S_w[i-1,j] + S_w[i+1,j] + S_w[i,(j+1)%N] + S_w[i,j-1] ) / 4 + \
                        dt/8/dx*c.u_inj/c.phi * (
                                f_w(S_w[i - 1, j - 1], c) + 2 * f_w(S_w[i - 1, j], c) + f_w(S_w[i - 1, (j + 1)%N], c) +
                            -   f_w(S_w[i + 1, j - 1], c) - 2 * f_w(S_w[i + 1, j], c) - f_w(S_w[i + 1, (j + 1)%N], c)
                        )

    S_w = newS_w
    S_w_all.append(newS_w)


print(len(S_w))
plt.contourf(S_w)
plt.show()
plt.figure()
plt.plot(np.linspace(0,L,N),S_w[:,1])
plt.scatter(shockspeed*t_tot,0)
plt.show()

