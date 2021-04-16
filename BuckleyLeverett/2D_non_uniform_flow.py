import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
from reservoirModule import *
from scipy.sparse import  diags

# initial variables
dx = 0.2
W  = 2
L  = 10

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


# determine number of elements, note I add two elements on the left and right as dummies,
N = int(W/dx)+2
# I add one on the length, as I need to have the pressure for all indices, 0,dx,...,L-dx,L,
M = int(L/dx)+1
print("N = ",N)
print("M = ",M)

# set initial Sw
Sw = c.S_wc*np.ones(N*M)
Sw[0:N] = 1-c.S_or

# prealocate memory
l_tot  = np.zeros(N*M)
dl_tot = np.zeros(N*M)
dl_totdx = np.zeros(N*M)
dl_totdy = np.zeros(N*M)

# for all elements, calculate the following
K = N*M
for i in range(K):
    # calculate lambda_tot
    l_tot[i]  = l_t(Sw[i],c)
    dl_tot[i] = dl_w(Sw[i],c)+dl_o(Sw[i],c)
for i in range(K):
    # calculate position derivatives of l_tot,
    # since some BC are periodic they need to be calculated differently
    if i%N == N-1:
        dl_totdx[i] = 1 / 2 / dx * (dl_tot[(i + 1-N)] - dl_tot[(i - 1)])
    elif i%N == 0:
        dl_totdx[i] = 1 / 2 / dx * (dl_tot[(i + 1)] - dl_tot[(i - 1 + N)])
    else:
        dl_totdx[i] = 1 / 2 / dx * (dl_tot[(i + 1)] - dl_tot[(i - 1)])

    if math.floor(i/N) == M-1:
        dl_totdy[i] = 1 / dx * (dl_tot[(i)] - dl_tot[(i - N)])
    elif math.floor(i/N) == 0:
        # note that we have a less accurate calculation here
        dl_totdy[i] = 1 / dx * (dl_tot[i+N] - dl_tot[(i)])
    else:
        dl_totdy[i] = 1 / 2 / dx * (dl_tot[i + N] - dl_tot[i-N])


# set up diagonals
diagonals = [-4/dx/dx*l_tot,
             l_tot[0:-1]/dx/dx+dl_totdx[0:-1]/2/dx,
             l_tot[0:-1]/dx/dx-dl_totdx[0:-1]/2/dx,
             l_tot[0:-N]/dx/dx+dl_totdy[0:-N]/2/dx,
             l_tot[0:-N]/dx/dx-dl_totdy[0:-N]/2/dx]
D = diags(diagonals, [0,1,-1,N,-N]).toarray()

# create a list to remove the additional left and right elements (effectively we reduce N by 2)
list = []
for i in range(K):
    if i%N == 0:
        list.append(i)
    elif i%N == N-1:
        list.append(i)

# due to the periodic BC, we have to add the correct items in to the matrix (these are now in the
# extra left and right elements and after we remove these, we have to add them back in.
for j in range(M):
    index = j*(N) + (N-2)
    D[index, j * N + 1    ]  = (l_tot[index] / dx / dx + dl_totdx[index] / 2 / dx)
    index = j*(N) + 1
    D[index, j * N + N - 2]  = (l_tot[index] / dx / dx - dl_totdx[index] / 2 / dx)

# Since we have Neumann BC, we can eliminate the some of the values and need to add others:
# we can get rid of the p_y terms, but we have to use a different stencil for the laplacian.
for i in range(N):
    index = i
    D[index, index + N] = 2 * (l_tot[index] / dx / dx)
    index = i + (M-1)*(N)
    D[index, index - N] = 2 * (l_tot[index] / dx / dx)


# remove the additional columns and rows who belong to the additional left and right elements.
D = np.delete(D,list,axis=1)
D = np.delete(D,list,axis=0)

# since the system is determined up to a constant, we add a extra sum over all P and set that to zero.
l = np.ones((N-2)*(M))
D = np.vstack([D,l])

# create right hand side vector, set last element to zero for total sum
l = np.zeros(((N-2)*(M))+1)
l[-1] = 0

# only the first N-2 elements have inflow and the last N-2 due to outflow
for i in range((N-2)*M):
    if math.floor(i/(N-2)) == 0:
        index = i%(N-2)+1
        l[i] = (dl_totdy[index]+2/dx*l_tot[index])*c.u_inj/l_w(Sw[index],c)
    if math.floor(i/(N-2)) == M-1:
        index = (M-1)*N + i % (N - 2) + 1
        l[i] = (dl_totdy[index]+2/dx*l_tot[index])*c.u_inj/l_t(Sw[index],c)

# solve matrix vector product
p = np.linalg.lstsq(D,l, rcond=None)
p = p[0]

# plot
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface( z=p.reshape(M,N-2),x = np.linspace(0,W,N-2), y = np.linspace(0,L,M))])
fig.show()

