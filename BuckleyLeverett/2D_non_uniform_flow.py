import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
from reservoirModule import *
from Non_Uniform_functions import *
from scipy.sparse import  diags

# initial variables
dx = 0.5
W  = 2
L  = 10
dt = 0.005
t_tot = 0.05

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
    labda = 1,
    dx = 1)

c.dx = dx
# determine number of elements, note I add two elements on the left and right as dummies,
N = int(W/dx)+2
# I add one on the length, as I need to have the pressure for all indices, 0,dx,...,L-dx,L,
M = int(L/dx)+1
print("N = ",N)
print("M = ",M)

# set initial Sw
Sw = c.S_wc*np.ones((N-2)*M)
Sw[0:(N-2)] = 1-c.S_or
Sw = Sw.reshape(M,N-2)

# for t in tqdm.tqdm(range(int(t_tot/dt))):
#     p = calc_pressure(Sw,c)
#     Swt = calc_Swt(Sw,p,c)
#     Sw = Sw + dt*Swt
# for t in tqdm.tqdm(range(int(t_tot/dt))):
#     Sw0 = Sw
#     error = 0.1
#     while error >= 1e-3:
#         p = calc_pressure(Sw,c)
#         Swt = calc_Swt(Sw,p,c)
#         SwNew = Sw0 + dt*Swt
#         error = np.linalg.norm(SwNew-Sw)
#         Sw = SwNew


# Swt[0:N-2] = 0
p = calc_pressure(Sw,c)
Swt = calc_Swt(Sw,p,c)

import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface( z=p.reshape(M,N-2),x = np.linspace(0,W,N-2), y = np.linspace(0,L,M))])
fig.show()
# plot
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface( z=Swt,x = np.linspace(0,W,N-2), y = np.linspace(0,L,M))])
fig.show()

