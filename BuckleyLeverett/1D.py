import numpy as np
import matplotlib.pyplot as plt
import tqdm
from reservoirModule import *







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