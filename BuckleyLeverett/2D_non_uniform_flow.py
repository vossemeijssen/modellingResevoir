import numpy as np
import matplotlib.pyplot as plt
import tqdm
from reservoirModule import *

dx = 0.1
L  = 10
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
    S_wc = 0.1 )


N = int(L/dx)+1
M = int(W/dx)+1

Sw = np.zeros(N*M)
p  = np.zeros(N*M)

dl_tot = np.zeros(N*M)

for i in range(N*M):
    if i%
    xp1 = i + 1
    xm1
    dl_tot = dl_w(Sw[i],c)+dl_o(Sw[i],c)



