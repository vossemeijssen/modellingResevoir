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
dt = dx/shockspeed  # time step
k = 0.1
# Code
N = int(L/dx)
time_N = int(t_tot / dt)
S_w = np.ones((N,N)) * S_wc
for j in range(N):
    M = 1+int(5*(1+np.sin(j*k)))
    for i in range(M):
        S_w[i,j] = 1 - S_or
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
    for j in range(1, N - 1):
        M = 1 + int(5 * (1 + np.sin(j * k)))
        for i in range(M+1, N-1):


            # implementation of Lax–Friedrichs Method
            newS_w[i] = ( S_w[i-1,j-1] + S_w[i+1,j-1] + S_w[i-1,j+1] + S_w[i+1,j+1] ) / 4 + \
                        dt/6/dx *u_inj/phi *(
                                f_w(S_w[i - 1, j - 1]) + 2 * f_w(S_w[i - 1, j]) + f_w(S_w[i - 1, j + 1]) +
                            -   f_w(S_w[i + 1, j - 1]) - 2 * f_w(S_w[i + 1, j]) - f_w(S_w[i + 1, j + 1])
                        )

    S_w = newS_w
    S_w_all.append(newS_w)


print(len(S_w))
#plt.contourf(S_w)
plt.plot(np.linspace(0,L,N),S_w[:,1])
#plt.scatter(shockspeed*t_tot,0)
plt.show()

