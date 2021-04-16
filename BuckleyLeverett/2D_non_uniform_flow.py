import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
from reservoirModule import *
from scipy.sparse import  diags

dx = 0.5
L  = 6
W  = 10

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
    sigma = 1,
    labda = 1)


N = int(L/dx)+2
M = int(W/dx)+1
print("N = ",N)
print("M = ",M)


Sw = c.S_wc*np.ones(N*M)
Sw[0:N] = 1-c.S_or
# print(Sw.reshape(M,N))
p  = np.zeros(N*M)

l_tot  = np.zeros(N*M)
dl_tot = np.zeros(N*M)
dl_totdx = np.zeros(N*M)
dl_totdy = np.zeros(N*M)

K = N*M
for i in range(K):
    l_tot[i]  = l_t(Sw[i],c)
    dl_tot[i] = dl_w(Sw[i],c)+dl_o(Sw[i],c)
for i in range(K):
    if i%N == N-1:
        dl_totdx[i] = 1 / 2 / dx * (dl_tot[(i + 1-N)] - dl_tot[(i - 1)])
    elif i%N == 0:
        dl_totdx[i] = 1 / 2 / dx * (dl_tot[(i + 1)] - dl_tot[(i - 1 + N)])
    else:
        dl_totdx[i] = 1 / 2 / dx * (dl_tot[(i + 1)] - dl_tot[(i - 1)])

    if math.floor(i/N) == M-1:
        dl_totdy[i] = 1 / dx * (dl_tot[(i)] - dl_tot[(i - N)])
    elif math.floor(i/N) == 0:
        dl_totdy[i] = 1 / dx * (dl_tot[i+N] - dl_tot[(i)])
    else:
        dl_totdy[i] = 1 / 2 / dx * (dl_tot[i + N] - dl_tot[i-N])



diagonals = [-4/dx/dx*l_tot,
             l_tot[0:-1]/dx/dx+dl_totdx[0:-1]/2/dx,
             l_tot[0:-1]/dx/dx-dl_totdx[0:-1]/2/dx,
             l_tot[0:-N]/dx/dx+dl_totdy[0:-N]/2/dx,
             l_tot[0:-N]/dx/dx-dl_totdy[0:-N]/2/dx]
D = diags(diagonals, [0,1,-1,N,-N]).toarray()

list = []
for i in range(K):
    if i%N == 0:
        list.append(i)
    elif i%N == N-1:
        list.append(i)
    # elif math.floor(i/N) == 0:
    #     list.append(i)
    # elif math.floor(i / N) == M-1:
    #     list.append(i)


for j in range(M):
    index = j*(N) + (N-2)
    D[j * N + 1    , index]  = (l_tot[index] / dx / dx + dl_totdx[index] / 2 / dx)
    index = j*(N) + 1
    D[j * N + N - 2, index]  = (l_tot[index] / dx / dx - dl_totdx[index] / 2 / dx)

for i in range(N):
    index = i
    D[index + N, index] = 2 * (l_tot[index] / dx / dx)
    index = i + (M-1)*(N)
    D[index - N, index] = 2 * (l_tot[index] / dx / dx)

# print(np.shape(D))
# print(list)
# plt.matshow(D)

D = np.delete(D,list,axis=1)
D = np.delete(D,list,axis=0)

# print(np.shape(D))
# plt.matshow(D)
# plt.show()
# print(np.shape(D))
l = np.ones((N-2)*(M))
l[-1] = 1
D = np.vstack([D,l])
# print(D)
l = np.zeros(((N-2)*(M))+1)
l[-1] = 0

for i in range((N-2)*M):
    if math.floor(i/(N-2)) == 0:
        index = i%(N-2)+1
        l[i] = (dl_totdy[index]+2/dx)*c.u_inj/l_w(Sw[index],c)
    if math.floor(i/(N-2)) == M-1:
        index = (M-1)*N + i % (N - 2) + 1
        l[i] = (dl_totdy[index]+2/dx)*c.u_inj/l_t(Sw[index],c)


p = np.linalg.lstsq(D,l, rcond=None)
p = p[0]
# print(p.reshape(M-2,N-2))
#
# plt.contourf(p.reshape(M,N-2))
# plt.matshow(Sw.reshape(M,N))
# plt.show()
plt.matshow(D)
plt.show()

import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(x = np.linspace(0,W,M), y = np.linspace(0,L,N-2), z=p.reshape(M,N-2))])
fig.show()

