import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
from reservoirModule import *
from scipy.sparse import  diags

dx = int(5)
L  = int(20)
W  = int(20)

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


N = int(L/dx)+1
M = int(W/dx)+1

Sw = np.zeros(N*M)
p  = np.zeros(N*M)

dl_tot = np.zeros(N*M)
dl_totdx = np.zeros(N*M)
dl_totdy = np.zeros(N*M)

K = N*M
for i in range(K):
    dl_tot[i] = dl_w(Sw[i],c)+dl_o(Sw[i],c)
for i in range(K):
    dl_totdx[i] = 1 / 2 / dx * (dl_tot[(i + 1    ) % K] - dl_tot[(i - 1    ) % K])
    dl_totdy[i] = 1 / 2 / dx * (dl_tot[(i + N + 1) % K] - dl_tot[(i - N - 1) % K])



diagonals = [-4/dx/dx*dl_tot,
             dl_tot[0:-1]/dx/dx+dl_totdx[0:-1]/2/dx,
             dl_tot[0:-1]/dx/dx-dl_totdx[0:-1]/2/dx,
             dl_tot[0:-N]/dx/dx+dl_totdy[0:-N]/2/dx,
             dl_tot[0:-N]/dx/dx-dl_totdy[0:-N]/2/dx]
D = diags(diagonals, [0,1,-1,N,-N]).toarray()

list = []
for i in range(K):
    if i%N == 0:
        list.append(i)
    if i%N == N-1:
        list.append(i)
    if math.floor(i/N) == 0:
        list.append(i)
    if math.floor(i/N) == N-1:
        list.append(i)

D = np.delete(D,list,axis=1)
D = np.delete(D,list,axis=0)
print(D)
# plt.show(?)


