import numpy as np
import matplotlib.pyplot as plt
import tqdm
from reservoirModule import *

# Variables
L = 10  # Total length
W = 1   # Total Width
dx = 0.1  # distance step
t_tot = 0.3  # Total time
# initial disturbance
k = 1               # number of waves
Amplitude = 0.5       # amplitude in meters

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


N = int(L/dx)
M = int(W/dx)
time_N = int(t_tot / dt)
S_w = np.ones((N,M)) * c.S_wc
for j in range(M):
    M_BC = 3+int(Amplitude/dx*(1+np.cos(2*np.pi*k*j*dx/W)))
    for i in range(M_BC):
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
    for j in range(0, M):
        M_BC = 3+int(Amplitude/dx*(1+np.cos(2*np.pi*k*j*dx/W)))
        for i in range(M_BC, N-1):
            # implementation of Lax–Friedrichs Method
            #newS_w[i,j] = ( S_w[i-1,j-1] + S_w[i+1,j-1] + S_w[i-1,(j+1)%M] + S_w[i+1,(j+1)%M] ) / 4 + \
            #            dt/16/dx*c.u_inj/c.phi * (
            #                    f_w(S_w[i - 1, j - 1], c) + 6 * f_w(S_w[i - 1, j], c) + f_w(S_w[i - 1, (j + 1)%M], c) +
            #                -   f_w(S_w[i + 1, j - 1], c) - 6 * f_w(S_w[i + 1, j], c) - f_w(S_w[i + 1, (j + 1)%M], c)
            #            )
            ## total x diffusion beun
            newS_w[i, j] = (4 * S_w[i - 1, j    ] +     S_w[i - 1, (j + 1) % M] + S_w[i - 1, j - 1] +
                            4 * S_w[i + 1, j    ] +     S_w[i + 1, (j + 1) % M] + S_w[i + 1, j - 1] +
                            4 * S_w[i    , j - 1] + 4 * S_w[i    , (j + 1) % M]) / 20 + \
                           dt / 16 / dx * c.u_inj / c.phi * (
                                   f_w(S_w[i - 1, j - 1], c) + 6 * f_w(S_w[i - 1, j], c) + f_w(S_w[i - 1, (j + 1) % M], c) +
                               -   f_w(S_w[i + 1, j - 1], c) - 6 * f_w(S_w[i + 1, j], c) - f_w(S_w[i + 1, (j + 1) % M], c)
                           )
            ## beun directional diffusion
            # newS_w[i, j] = (8 * S_w[i - 1, j] + S_w[i - 1, (j + 1) % M] + S_w[i - 1, j - 1] +
            #                 8 * S_w[i + 1, j] + S_w[i + 1, (j + 1) % M] + S_w[i + 1, j - 1] +
            #                 0 * S_w[i, j - 1] + 0 * S_w[i, (j + 1) % M]) / 20 + \
            #                dt / 16 / dx * c.u_inj / c.phi * (
            #                        f_w(S_w[i - 1, j - 1], c) + 6 * f_w(S_w[i - 1, j], c) + f_w(S_w[i - 1, (j + 1) % M],
            #                                                                                    c) +
            #                        -   f_w(S_w[i + 1, j - 1], c) - 6 * f_w(S_w[i + 1, j], c) - f_w(
            #                    S_w[i + 1, (j + 1) % M], c)
            #                )
            ## simple diffusion
            # newS_w[i, j] = (S_w[i-1,j] + S_w[i+1,j]) / 2 + \
            #                dt / 16 / dx * c.u_inj / c.phi * (
            #                        f_w(S_w[i - 1, j - 1], c) + 6 * f_w(S_w[i - 1, j], c) + f_w(S_w[i - 1, (j + 1) % M],
            #                                                                                    c) +
            #                        -   f_w(S_w[i + 1, j - 1], c) - 6 * f_w(S_w[i + 1, j], c) - f_w(
            #                    S_w[i + 1, (j + 1) % M], c)
            #                )

    S_w = newS_w
    S_w_all.append(newS_w)


print(len(S_w))
plt.contourf(S_w)
plt.show()
plt.figure()
plt.plot(np.linspace(0,L,N),S_w[:,0])
plt.scatter(2*Amplitude+shockspeed*t_tot,0)
plt.show()


# fig = plt.figure()
# X = np.arange(0, N)
# Y = np.arange(0, N)
# X, Y = np.meshgrid(X, Y)
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(Y, X, S_w, rstride=1, cmap="rainbow", cstride=1, antialiased=False)
# fig.colorbar(surf)
# plt.show()
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=S_w)])

# fig.update_layout(title='Saturation after ' + str(t_tot) + "s", autosize=True,
#                   width=N, height=N,
#                   margin=dict(l=65, r=50, b=65, t=90))

fig.show()
