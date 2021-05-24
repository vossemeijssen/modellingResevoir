from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
from reservoirModule import *
from scipy.sparse.linalg import isolve
from scipy import sparse
from timeit import timeit
import scipy
import random



hx = 0.25
hy = 0.25
W  = 2
L  = 3

kNum = 1
delta = 0.001

beun_factor  = 200

plotting = False

t_end = 0.1

eps = 1e-9

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

phi = 0.1 # Porosity
u_inj = 1.0 # Water injection speed
mu_w = 1.e-3
mu_o = 0.04
kappa = 1.0
k_rw0 = 1.0
k_ro0 = 1.0
n_w = 4.0  # >=1
n_o = 2.0  # >=1
S_or = 0.1  # Oil rest-saturation
S_wc = 0.1
sigma = 1.0
labda = 1.0
dx = 1.0

N = int(W / hx)
M = int(L / hy)

@jit(nopython=True)
def limiter(phi):
    if np.isnan(phi):
        return 0
    else:
        return 0+0*(phi+abs(phi))/(1+abs(phi))

@jit(nopython=True)
def NanDiff(a,b):
    if b == 0:
        return np.nan
    else:
        return a/b

@jit(nopython=True)
def l_w(S_w):
    S_wn = (S_w - S_wc) / (1.0 - S_or - S_wc)
    return kappa / (mu_w) * k_rw0 * (S_wn) ** n_w

@jit(nopython=True)
def l_o(S_w):
    S_wn = (S_w - S_wc) / (1.0 - S_or - S_wc)
    return kappa / mu_o * k_ro0 *  (1-S_wn) ** n_o

@jit(nopython=True)
def l_t(S_w):
    return l_w(S_w) + l_o(S_w)


def magic_function(x, c):
    return df_dSw(x, c) - (f_w(x, c) - f_w(c.S_wc, c))/(x - c.S_wc)

S_w_shock = bisection(magic_function, (c.S_wc, 1 - c.S_or), 100, c)
shockspeed = c.u_inj/c.phi*df_dSw(S_w_shock, c)

# why the constant 6?
dt = min(hx,hy)/shockspeed/beun_factor
Tstep = int(t_end/dt)

print("shockspeed = ",shockspeed,", shock position = ",shockspeed*dt*Tstep,", maximum dt = ",min(hx,hy)/shockspeed/2)


Sw = 0.1*np.ones(N*(M+1) + (N+1)*M)
for i in range(N):
    Sw[i] = 0.9



@jit(nopython=True)
def FVM_pressure_Mat(Sw,N,M,hx,hy):
    # set up S matrix for each volume, and each pressure
    S = np.zeros( (N*M+1, N*M))


    i = 0; j = 0
    index_p = i + j * N
    index_Sw_xP = N * (M + 1) + i + j * (N + 1)
    index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
    index_Sw_yN = i + (j + 1) * N
    index_Sw_yP = i + j * N

    S[index_p, index_p] = -hy / hx * l_t(Sw[index_Sw_xN]) - hx / hy * l_t(Sw[index_Sw_yN])
    S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN])
    S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN])

    i = 0; j = M - 1
    index_p = i + j * N
    index_Sw_xP = N * (M + 1) + i + j * (N + 1)
    index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
    index_Sw_yN = i + (j + 1) * N
    index_Sw_yP = i + j * N

    S[index_p, index_p] = -hy / hx * l_t(Sw[index_Sw_xN]) - hx / hy * l_t(Sw[index_Sw_yP])
    S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN])
    S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP])

    i = N - 1; j = 0
    index_p = i + j * N
    index_Sw_xP = N * (M + 1) + i + j * (N + 1)
    index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
    index_Sw_yN = i + (j + 1) * N
    index_Sw_yP = i + j * N

    S[index_p, index_p] = - hy / hx * l_t(Sw[index_Sw_xP]) - hx / hy * l_t(Sw[index_Sw_yN])
    S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP])
    S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN])

    i = N - 1; j = M - 1
    index_p = i + j * N
    index_Sw_xP = N * (M + 1) + i + j * (N + 1)
    index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
    index_Sw_yN = i + (j + 1) * N
    index_Sw_yP = i + j * N

    S[index_p, index_p] = - hy / hx * l_t(Sw[index_Sw_xP]) - hx / hy * l_t(Sw[index_Sw_yP])
    S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP])
    S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP])


    # i = 0 and i == N - 1
    for j in range(1,M-1):
        i = 0
        index_p = i + j * N
        index_Sw_xP = N * (M + 1) + i + j * (N + 1)
        index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
        index_Sw_yN = i + (j + 1) * N
        index_Sw_yP = i + j * N

        S[index_p, index_p] = -hy / hx * l_t(Sw[index_Sw_xN]) - hx / hy * l_t(Sw[index_Sw_yN]) - hx / hy * l_t(Sw[index_Sw_yP])
        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN])
        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN])
        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP])

        i = N - 1
        index_p = i + j * N
        index_Sw_xP = N * (M + 1) + i + j * (N + 1)
        index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
        index_Sw_yN = i + (j + 1) * N
        index_Sw_yP = i + j * N

        S[index_p, index_p] = - hy / hx * l_t(Sw[index_Sw_xP]) - hx / hy * l_t(Sw[index_Sw_yN]) - hx / hy * l_t(Sw[index_Sw_yP])
        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP])
        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN])
        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP])

    # for j = 0 and j = M - 1
    for i in range(1,N-1):
        j = 0
        index_p = i + j * N
        index_Sw_xP = N * (M + 1) + i + j * (N + 1)
        index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
        index_Sw_yN = i + (j + 1) * N
        index_Sw_yP = i + j * N

        S[index_p, index_p] = -hy / hx * l_t(Sw[index_Sw_xN]) - hy / hx * l_t(Sw[index_Sw_xP]) - hx / hy * l_t(Sw[index_Sw_yN])
        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN])
        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP])
        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN])

        j = M - 1
        index_p = i + j * N
        index_Sw_xP = N * (M + 1) + i + j * (N + 1)
        index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
        index_Sw_yN = i + (j + 1) * N
        index_Sw_yP = i + j * N

        S[index_p, index_p] = -hy / hx * l_t(Sw[index_Sw_xN]) - hy / hx * l_t(Sw[index_Sw_xP]) - hx / hy * l_t(Sw[index_Sw_yP])
        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN])
        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP])
        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP])

    for i in range(1,N-1):
        for j in range(1,M-1):
            index_p = i + j * N
            index_Sw_xP = N * (M + 1) + i + j * (N + 1)
            index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
            index_Sw_yN = i + (j + 1) * N
            index_Sw_yP = i + j * N

            S[index_p, index_p] = -hy / hx * l_t(Sw[index_Sw_xN]) - hy / hx * l_t(Sw[index_Sw_xP]) - hx / hy * l_t( Sw[index_Sw_yN]) - hx / hy * l_t(Sw[index_Sw_yP])
            S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN])
            S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP])
            S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN])
            S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP])

    f = np.zeros(N*M+1)
    for i in range(N):
        for j in range(M):
            index_p = i + j * N
            if j == 0:
                f[index_p] = -hx*u_inj
            elif j == M-1:
                f[index_p] = hx*u_inj

    l = np.zeros(N*M)
    l[-1] = 1.0
    # S = np.vstack((S,l))
    S[N*M,N*M-1] = 1.0

    # p = np.linalg.lstsq(S,f,rcond=None)
    return S,f

def FVM_pressure(Sw,N,M,hx,hy):
    S, f = FVM_pressure_Mat(Sw,N,M,hx,hy)
    p = np.linalg.lstsq(S,f,rcond=None)
    # p = scipy.sparse.linalg.lsqr(S,)
    return p[0]

def FVM_pressure_CG(pOld,Sw,N,M,hx,hy):
    S, f = FVM_pressure_Mat(Sw, N, M, hx, hy)
    St   = np.transpose(S)
    StS = np.dot(St,S)
    Stf = np.dot(St,f)
    # p = isolve.cg(StS,Stf,pOld,eps)
    # p = isolve.bicgstab(StS,Stf,pOld,eps)
    p = scipy.sparse.linalg.lsqr(S,f,x0 = pOld)
    return p[0]

def FVM_Sw_t(p,Sw,N,M,hx,hy):
    Swt = np.zeros(N*(M+1) + (N+1)*M)


    for j in range(M):
        for i in range(N+1):
            index_Swt   = N * (M + 1) + i + (N + 1) * j

            index_p     = i - 1 + j * N
            index_Sw_yP = i - 1 + j * N

            boundary1 = 0
            boundary2 = 0
            boundary3 = 0
            boundary4 = 0

            ## vertical flows
            if i == 0:
                if j == 0:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p + 1 + N] - p[index_p + 1])/hy
                    boundary4 = u_inj*hx/2
                elif j == 1:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p + 1 + N] - p[index_p + 1])/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ]) * (p[index_p + 1 - N] - p[index_p + 1])/hy
                elif j == M-1:
                    boundary3 = 0
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ]) * (p[index_p + 1 - N] - p[index_p + 1])/hy
                else:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p + 1 + N    ] - p[index_p + 1 - N])/2/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ]) * (p[index_p + 1 - 2 * N] - p[index_p + 1    ])/2/hy
            elif i == N:
                if j == 0:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N]) * (p[index_p     + N] - p[index_p    ])/hy
                    boundary4 = u_inj*hx/2
                elif j == 1:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N]) * (p[index_p     + N] - p[index_p    ])/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ]) * (p[index_p     - N] - p[index_p    ])/hy
                elif j == M-1:
                    boundary3 = 0
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ]) * (p[index_p     - N] - p[index_p    ])/hy
                else:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N]) * (p[index_p     + N] - p[index_p - N])/2/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ]) * (p[index_p - 2 * N] - p[index_p    ])/2/hy
            else:
                if j==0:
                    boundary3 = hx*(l_w(Sw[index_Sw_yP + N])+l_w(Sw[index_Sw_yP + N + 1]))/2 * (p[index_p+N] + p[index_p+1+N] - p[index_p] - p[index_p+1])/2/hy
                    boundary4 = u_inj*hx
                elif j == 1:
                    boundary3 = hx*(l_w(Sw[index_Sw_yP + N])+l_w(Sw[index_Sw_yP + N + 1]))/2 * (p[index_p+N] + p[index_p+1+N] - p[index_p] - p[index_p+1])/2/hy
                    boundary4 = hx*(l_w(Sw[index_Sw_yP    ])+l_w(Sw[index_Sw_yP + 1    ]))/2 * (p[index_p-N] + p[index_p+1-N] - p[index_p] - p[index_p+1])/2/hy
                elif j==M-1:
                    boundary3 = 0
                    boundary4 = hx*(l_w(Sw[index_Sw_yP    ])+l_w(Sw[index_Sw_yP + 1    ]))/2 * (p[index_p-N] + p[index_p+1-N] - p[index_p] - p[index_p+1])/2/hy
                else:
                    boundary3 = hx*(l_w(Sw[index_Sw_yP + N])+l_w(Sw[index_Sw_yP + N + 1]))/2 * (p[index_p + N] + p[index_p+1+N  ] - p[index_p-N] - p[index_p+1-N])/4/hy
                    boundary4 = hx*(l_w(Sw[index_Sw_yP    ])+l_w(Sw[index_Sw_yP + 1    ]))/2 * (p[index_p-2*N] + p[index_p+1-2*N] - p[index_p  ] - p[index_p+1  ])/4/hy

            ## horizonatal flows
            if i==0:
                boundary1 = 0
                boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p + 1]) / hx
            elif i==N:
                boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p    ]) / hx
                boundary2 = 0
            else:
                boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p    ] - p[index_p + 1]) / hx
                boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 1] - p[index_p    ]) / hx

            Swt[index_Swt] = (boundary1 + boundary2 + boundary3 + boundary4)/phi/hx/hy
            if i == 0 or i == N:
                Swt[index_Swt] = Swt[index_Swt]*2



    for j in range(M+1):
        for i in range(N):
            index_Swt=  i + N * j

            index_p     = i + (j - 1) * N
            index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

            boundary1 = 0
            boundary2 = 0
            boundary3 = 0
            boundary4 = 0

            ## horizontal flow
            if j == 0:
                continue
                if i == 0:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1]) * (2*p[index_p+1+N]+hx*u_inj/l_w(Sw[index_Swt+1])-2*p[index_p+N]-hx*u_inj/l_w(Sw[index_Swt]))/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP     + N + 1]) * (2*p[index_p+N]+hx*u_inj/l_w(Sw[index_Swt])-2*p[index_p - 1 + N]-hx*u_inj/l_w(Sw[index_Swt-1]))/hx
                else:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1]) * (2*p[index_p+1+N]+hx*u_inj/l_w(Sw[index_Swt+1])-2*p[index_p+N]-hx*u_inj/l_w(Sw[index_Swt]))/hx
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP     + N + 1]) * (2*p[index_p+N]+hx*u_inj/l_w(Sw[index_Swt])-2*p[index_p - 1 + N]-hx*u_inj/l_w(Sw[index_Swt-1]))/hx
            elif j == M:
                if i == 0:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1    ]) * (p[index_p + 1]-p[index_p])/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP + 1    ]) * (p[index_p - 1]-p[index_p])/hx
                else:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP+1])*(p[index_p+1]-p[index_p])/hx
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP+1])*(p[index_p-1]-p[index_p])/hx
            else:
                if i == 0:
                    boundary1 = hy*(l_w(Sw[index_Sw_xP + 1]) + l_w(Sw[index_Sw_xP + 1 + N + 1]))/2*(p[index_p + 1] + p[index_p + 1 + N] - p[index_p] - p[index_p + N])/2/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy*(l_w(Sw[index_Sw_xP    ]) + l_w(Sw[index_Sw_xP     + N + 1]))/2*(p[index_p - 1] + p[index_p - 1 + N] - p[index_p] - p[index_p + N])/2/hx
                else:
                    boundary1 = hy*(l_w(Sw[index_Sw_xP + 1]) + l_w(Sw[index_Sw_xP + 1 + N + 1]))/2*(p[index_p + 1] + p[index_p + 1 + N] - p[index_p] - p[index_p + N])/2/hx
                    boundary2 = hy*(l_w(Sw[index_Sw_xP    ]) + l_w(Sw[index_Sw_xP     + N + 1]))/2*(p[index_p - 1] + p[index_p - 1 + N] - p[index_p] - p[index_p + N])/2/hx

            ## vertical flow
            if j == 0:
                continue
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + N    ]-p[index_p    ])/hy
                boundary4 = u_inj*hx
            if j == 1:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2 * u_inj / l_w(Sw[index_Swt - N])   # estimate pressure difference with inflow pressure difference
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p] + hy * u_inj / l_w(Sw[index_Swt - N])  -p[index_p + N])/2/hy
            elif j == M:
                boundary3 = 0
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p    ])/hy
            elif j == M-1:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + N    ]-p[index_p    ])/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p + N])/2/hy
            else:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p + N])/2/hy

            Swt[index_Swt] = (boundary1 + boundary2 + boundary3 + boundary4)/phi/hx/hy
            if j == 0 or j == M:
                Swt[index_Swt] = Swt[index_Swt]*2
    return Swt

def FVM_Sw_t_2(p,Sw,N,M,hx,hy):
    Swt = np.zeros(N*(M+1) + (N+1)*M)


    for j in range(M):
        for i in range(N+1):
            index_Swt   = N * (M + 1) + i + (N + 1) * j

            index_p     = i - 1 + j * N
            index_Sw_yP = i - 1 + j * N

            boundary1 = 0
            boundary2 = 0
            boundary3 = 0
            boundary4 = 0

            ## vertical flows
            if i == 0:
                if j == 0:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p + 1 + N] - p[index_p + 1])/hy
                    boundary4 = u_inj*hx/2
                elif j == M-1:
                    boundary3 = 0
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ]) * (p[index_p + 1 - N] - p[index_p + 1])/hy
                else:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p + 1 + N] - p[index_p + 1])/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ]) * (p[index_p + 1 - N] - p[index_p + 1])/hy
            elif i == N:
                if j == 0:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N]) * (p[index_p + N] - p[index_p])/hy
                    boundary4 = u_inj*hx/2
                elif j == M-1:
                    boundary3 = 0
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ]) * (p[index_p - N] - p[index_p])/hy
                else:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N]) * (p[index_p + N] - p[index_p])/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ]) * (p[index_p - N] - p[index_p])/hy
            else:
                if j==0:
                    boundary3 = hx/2 * ( l_w(Sw[index_Sw_yP + N])* (p[index_p+N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p+N+1] - p[index_p + 1])/hy)
                    boundary4 = u_inj*hx
                elif j==M-1:
                    boundary3 = 0
                    boundary4 = hx/2 * ( l_w(Sw[index_Sw_yP + 0])* (p[index_p-N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + 0 + 1]) * (p[index_p-N+1] - p[index_p + 1])/hy)
                else:
                    boundary3 = hx/2 * ( l_w(Sw[index_Sw_yP + N])* (p[index_p+N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p+N+1] - p[index_p + 1])/hy)
                    boundary4 = hx/2 * ( l_w(Sw[index_Sw_yP + 0])* (p[index_p-N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + 0 + 1]) * (p[index_p-N+1] - p[index_p + 1])/hy)

            ## horizonatal flows
            if i == 0:
                boundary1 = 0
                boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p + 1]) / 2 / hx
            elif i == 1:
                boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p    ] - p[index_p + 1]) / 2 / hx
                boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p    ]) / 2 / hx
            elif i == N-1:
                boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p + 1]) / 2 / hx
                boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 1] - p[index_p    ]) / 2 / hx
            elif i == N:
                boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p    ]) / 2 / hx
                boundary2 = 0
            else:
                boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p + 1]) / 2 / hx
                boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p    ]) / 2 / hx

            Swt[index_Swt] = (boundary1 + boundary2 + boundary3 + boundary4)/phi/hx/hy
            if i == 0 or i == N:
                Swt[index_Swt] = Swt[index_Swt]*2



    for j in range(M+1):
        if j == 0:
            continue
        for i in range(N):
            index_Swt=  i + N * j

            index_p     = i + (j - 1) * N
            index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

            boundary1 = 0
            boundary2 = 0
            boundary3 = 0
            boundary4 = 0

            ## horizontal flow
            if j == 0:
                if i == 0:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1]) * (p[index_p + N + 1] - p[index_p + N])/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP + 0 + N + 1]) * (p[index_p + N - 1] - p[index_p + N])/hx
                else:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1]) * (p[index_p + N + 1] - p[index_p + N])/hx
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP + 0 + N + 1]) * (p[index_p + N - 1] - p[index_p + N])/hx
            elif j == M:
                if i == 0:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP+1])*(p[index_p+1]-p[index_p])/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP+0])*(p[index_p-1]-p[index_p])/hx
                else:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP+1])*(p[index_p+1]-p[index_p])/hx
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP+0])*(p[index_p-1]-p[index_p])/hx
            else:
                if i == 0:
                    boundary1 = hy/2 * (l_w(Sw[index_Sw_xP + 1])*(p[index_p + 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 2])*(p[index_p + N + 1]- p[index_p + N])/hx)
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2 * (l_w(Sw[index_Sw_xP    ])*(p[index_p - 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 1])*(p[index_p + N - 1]- p[index_p + N])/hx)
                else:
                    boundary1 = hy/2 * (l_w(Sw[index_Sw_xP + 1])*(p[index_p + 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 2])*(p[index_p + N + 1]- p[index_p + N])/hx)
                    boundary2 = hy/2 * (l_w(Sw[index_Sw_xP    ])*(p[index_p - 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 1])*(p[index_p + N - 1]- p[index_p + N])/hx)

            ## vertical flow
            if j == 0:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p + N] - hy*u_inj/l_w(Sw[index_Swt]))/2/hy
                boundary4 = u_inj*hx
            elif j == 1:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p        ]+ hy*u_inj/l_w(Sw[index_Swt-N])-p[index_p + N])/2/hy
            elif j == M:
                boundary3 = 0
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p    ])/2/hy
            elif j == M-1:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 1 * N]-p[index_p    ])/2/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p + N])/2/hy
            else:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p + N])/2/hy

            Swt[index_Swt] = (boundary1 + boundary2 + boundary3 + boundary4)/phi/hx/hy
            if j == 0 or j == M:
                Swt[index_Swt] = Swt[index_Swt]*2
    return Swt


def FVM_Sw_t_4(p,Sw,N,M,hx,hy):
    Swt = np.zeros(N*(M+1) + (N+1)*M)


    for j in range(M):
        for i in range(N+1):
            index_Swt   = N * (M + 1) + i + (N + 1) * j

            index_p     = i - 1 + j * N
            index_Sw_yP = i - 1 + j * N

            boundary1 = 0
            boundary2 = 0
            boundary3 = 0
            boundary4 = 0

            ## vertical flows
            if i == 0:
                if j == 0:
                    boundary4l = u_inj*hx/2
                    dpress3 = (p[index_p + N + 1] - p[index_p + 1]) / 2 / hy
                    if  dpress3 < 0: # positive flow, so upwind
                        boundary3l = hx / 2 * l_w(Sw[index_Swt]) * dpress3
                    else:
                        boundary3l = hx / 2 * l_w(Sw[index_Swt + N + 1]) * dpress3

                    boundary3h = hx/2*l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p + 1 + N] - p[index_p + 1])/hy
                    boundary4h = u_inj*hx/2
                elif j == M-1:
                    boundary3l = 0
                    dpress4 = (p[index_p + 1] - p[index_p - N + 1]) / 2 / hy
                    if dpress4 < 0:
                        boundary4l = -hx / 2 * l_w(Sw[index_Swt - N - 1]) * dpress4
                    else:
                        boundary4l = -hx / 2 * l_w(Sw[index_Swt]) * dpress4

                    boundary3h = 0
                    boundary4h = hx/2*l_w(Sw[index_Sw_yP + 1    ]) * (p[index_p + 1 - N] - p[index_p + 1])/hy
                else:
                    dpress3 = (p[index_p + N + 1] - p[index_p + 1]) / 2 / hy
                    dpress4 = (p[index_p + 1] - p[index_p - N + 1]) / 2 / hy
                    if  dpress3 < 0:
                        boundary3l = hx / 2 * l_w(Sw[index_Swt]) * dpress3
                    else:
                        boundary3l = hx / 2 * l_w(Sw[index_Swt + N + 1]) * dpress3
                    if dpress4 < 0:
                        boundary4l = -hx / 2 * l_w(Sw[index_Swt - N - 1]) * dpress4
                    else:
                        boundary4l = -hx / 2 * l_w(Sw[index_Swt]) * dpress4

                    boundary3h = hx/2*l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p + 1 + N] - p[index_p + 1])/hy
                    boundary4h = hx/2*l_w(Sw[index_Sw_yP + 1    ]) * (p[index_p + 1 - N] - p[index_p + 1])/hy
            elif i == N:
                if j == 0:
                    boundary4l = u_inj*hx/2
                    dpress3 = (p[index_p + N] - p[index_p]) / 2 / hy
                    if  dpress3 < 0:
                        boundary3l = hx / 2 * l_w(Sw[index_Swt]) * dpress3
                    else:
                        boundary3l = hx / 2 * l_w(Sw[index_Swt + N + 1]) * dpress3
                    boundary3h = hx/2*l_w(Sw[index_Sw_yP + N]) * (p[index_p + N] - p[index_p])/hy
                    boundary4h = u_inj*hx/2
                elif j == M-1:
                    boundary3l = 0
                    dpress4 = (p[index_p] - p[index_p - N]) / 2 / hy
                    if dpress4 < 0:
                        boundary4l = -hx / 2 * l_w(Sw[index_Swt - N - 1]) * dpress4
                    else:
                        boundary4l = -hx / 2 * l_w(Sw[index_Swt]) * dpress4
                    boundary3h = 0
                    boundary4h = hx/2*l_w(Sw[index_Sw_yP    ]) * (p[index_p - N] - p[index_p])/hy
                else:
                    dpress3 = (p[index_p + N] - p[index_p]) / 2 / hy
                    dpress4 = (p[index_p] - p[index_p - N]) / 2 / hy
                    if  dpress3 < 0:
                        boundary3l = hx / 2 * l_w(Sw[index_Swt]) * dpress3
                    else:
                        boundary3l = hx / 2 * l_w(Sw[index_Swt + N + 1]) * dpress3
                    if dpress4 < 0:
                        boundary4l = -hx / 2 * l_w(Sw[index_Swt - N - 1]) * dpress4
                    else:
                        boundary4l = -hx / 2 * l_w(Sw[index_Swt]) * dpress4

                    boundary3h = hx/2*l_w(Sw[index_Sw_yP + N]) * (p[index_p + N] - p[index_p])/hy
                    boundary4h = hx/2*l_w(Sw[index_Sw_yP    ]) * (p[index_p - N] - p[index_p])/hy
            else:
                if j==0:
                    dpress3 = (p[index_p + N] + p[index_p + N + 1] - p[index_p] - p[index_p + 1]) / 4 / hy
                    if  dpress3 < 0:
                        boundary3l = hx * l_w(Sw[index_Swt]) * dpress3
                    else:
                        boundary3l = hx * l_w(Sw[index_Swt + N + 1]) * dpress3
                    boundary4l = u_inj*hx

                    boundary3h = hx/2 * ( l_w(Sw[index_Sw_yP + N])* (p[index_p+N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p+N+1] - p[index_p + 1])/hy)
                    boundary4h = u_inj*hx
                elif j==M-1:
                    boundary3l = 0
                    dpress4 = (p[index_p] + p[index_p + 1] - p[index_p - N] - p[index_p - N + 1]) / 4 / hy
                    if dpress4 < 0:
                        boundary4l = -hx * l_w(Sw[index_Swt - N - 1]) * dpress4
                    else:
                        boundary4l = -hx * l_w(Sw[index_Swt]) * dpress4

                    boundary3h = 0
                    boundary4h = hx/2 * ( l_w(Sw[index_Sw_yP + 0])* (p[index_p-N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + 0 + 1]) * (p[index_p-N+1] - p[index_p + 1])/hy)
                else:
                    dpress3 = (p[index_p + N] + p[index_p + N + 1] - p[index_p] - p[index_p + 1]) / 4 / hy
                    dpress4 = (p[index_p] + p[index_p + 1] - p[index_p - N] - p[index_p - N + 1]) / 4 / hy
                    if  dpress3 < 0:
                        boundary3l = hx * l_w(Sw[index_Swt]) * dpress3
                    else:
                        boundary3l = hx * l_w(Sw[index_Swt + N + 1]) * dpress3
                    if dpress4 < 0:
                        boundary4l = -hx * l_w(Sw[index_Swt - N - 1]) * dpress4
                    else:
                        boundary4l = -hx * l_w(Sw[index_Swt]) * dpress4

                    boundary3h = hx/2 * ( l_w(Sw[index_Sw_yP + N])* (p[index_p+N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p+N+1] - p[index_p + 1])/hy)
                    boundary4h = hx/2 * ( l_w(Sw[index_Sw_yP + 0])* (p[index_p-N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + 0 + 1]) * (p[index_p-N+1] - p[index_p + 1])/hy)

            ## horizonatal flows
            if i == 0:
                boundary1l = 0
                dpress2    = (p[index_p + 2] - p[index_p + 1]) / 2 / hx
                if dpress2 < 0:
                    boundary2l = hy * l_w(Sw[index_Swt]) * dpress2
                else:
                    boundary2l = hy * l_w(Sw[index_Swt +  1]) * dpress2

                boundary1h = 0
                boundary2h = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p + 1]) / 2 / hx
            elif i == 1:
                dpress1 = (p[index_p    ] - p[index_p + 1]) / 2 / hx
                dpress2 = (p[index_p + 2] - p[index_p + 1]) / 2 / hx
                if dpress1 > 0:
                    boundary1l = hy * l_w(Sw[index_Swt - 1]) * dpress1
                else:
                    boundary1l = hy * l_w(Sw[index_Swt    ]) * dpress1
                if dpress2 < 0:
                    boundary2l = hy * l_w(Sw[index_Swt    ]) * dpress2
                else:
                    boundary2l = hy * l_w(Sw[index_Swt +  1]) * dpress2

                boundary1h = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p    ] - p[index_p + 1]) / 2 / hx
                boundary2h = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p    ]) / 2 / hx
            elif i == N-1:
                dpress1    = (p[index_p - 1] - p[index_p + 1]) / 2 / hx
                dpress2    = (p[index_p + 1] - p[index_p    ]) / 2 / hx
                if dpress1 > 0:
                    boundary1l = hy * l_w(Sw[index_Swt - 1]) * dpress1
                else:
                    boundary1l = hy * l_w(Sw[index_Swt    ]) * dpress1
                if dpress2 < 0:
                    boundary2l = hy * l_w(Sw[index_Swt    ]) * dpress2
                else:
                    boundary2l = hy * l_w(Sw[index_Swt +  1]) * dpress2

                boundary1h = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p + 1]) / 2 / hx
                boundary2h = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 1] - p[index_p    ]) / 2 / hx
            elif i == N:
                boundary2l = 0
                dpress1    = (p[index_p - 1] - p[index_p    ]) / 2 / hx
                if dpress1 > 0:
                    boundary1l = hy * l_w(Sw[index_Swt - 1]) * dpress1
                else:
                    boundary1l = hy * l_w(Sw[index_Swt]) * dpress1

                boundary1h = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p    ]) / 2 / hx
                boundary2h = 0
            else:
                dpress1    = (p[index_p - 1] - p[index_p + 1]) / 2 / hx
                dpress2    = (p[index_p + 2] - p[index_p    ]) / 2 / hx
                if dpress1 > 0:
                    boundary1l = hy * l_w(Sw[index_Swt - 1]) * dpress1
                else:
                    boundary1l = hy * l_w(Sw[index_Swt    ]) * dpress1
                if dpress2 < 0:
                    boundary2l = hy * l_w(Sw[index_Swt    ]) * dpress2
                else:
                    boundary2l = hy * l_w(Sw[index_Swt +  1]) * dpress2

                boundary1h = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p + 1]) / 2 / hx
                boundary2h = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p    ]) / 2 / hx


            NumElementsSw = np.size(Sw)

            u1 = (Sw[index_Sw_yP]+Sw[index_Sw_yP + N])/2
            r1 = NanDiff(u1 - Sw[index_Swt-1], Sw[index_Swt] - u1)
            boundary1 = boundary1l + limiter(r1)*(boundary1h - boundary1l)

            u2 = (Sw[index_Sw_yP + 1] + Sw[index_Sw_yP + N + 1]) / 2
            r2 = NanDiff(u2 - Sw[index_Swt], Sw[(index_Swt + 1)%NumElementsSw] - u2)
            boundary2 = boundary2l + limiter(r2) * (boundary2h - boundary2l)

            if i == 0:
                u3 = (Sw[index_Sw_yP + N + 1]) / 1
            elif i == N - 1:
                u3 = (Sw[index_Sw_yP + N]) / 1
            else:
                u3 = (Sw[index_Sw_yP + N] + Sw[index_Sw_yP + N + 1]) / 2
            r3 = NanDiff(u3 - Sw[index_Swt], Sw[(index_Swt + N + 1)%NumElementsSw] - u3)
            boundary3 = boundary3l + limiter(r3)*(boundary3h - boundary3l)

            if i == 0:
                u4 = (Sw[index_Sw_yP + 1]) / 1
            elif i == N - 1:
                u4 = (Sw[index_Sw_yP]) / 1
            else:
                u4 = (Sw[index_Sw_yP] + Sw[index_Sw_yP + 1]) / 2
            if index_Swt - N - 1 < N * (M+1):
                r4 = 0
            else:
                r4 = NanDiff(u4 - Sw[index_Swt - N - 1], Sw[index_Swt] - u4)
            boundary4 = boundary4l + limiter(r4)*(boundary4h - boundary4l)

            boundary1 = boundary1h
            boundary2 = boundary2h


            Swt[index_Swt] = (boundary1 + boundary2 + boundary3 + boundary4)/phi/hx/hy
            if i == 0 or i == N:
                Swt[index_Swt] = Swt[index_Swt]*2



    for j in range(M+1):
        if j == 0 or j == M:
            continue
        for i in range(N):
            index_Swt=  i + N * j

            index_p     = i + (j - 1) * N
            index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

            boundary1 = 0
            boundary2 = 0
            boundary3 = 0
            boundary4 = 0

            ## horizontal flow
            if j == 0:
                if i == 0:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1]) * (p[index_p + N + 1] - p[index_p + N])/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP + 0 + N + 1]) * (p[index_p + N - 1] - p[index_p + N])/hx
                else:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1]) * (p[index_p + N + 1] - p[index_p + N])/hx
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP + 0 + N + 1]) * (p[index_p + N - 1] - p[index_p + N])/hx
            elif j == M:
                if i == 0:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP+1])*(p[index_p+1]-p[index_p])/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP+0])*(p[index_p-1]-p[index_p])/hx
                else:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP+1])*(p[index_p+1]-p[index_p])/hx
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP+0])*(p[index_p-1]-p[index_p])/hx
            else:
                if i == 0:
                    boundary1 = hy/2 * (l_w(Sw[index_Sw_xP + 1])*(p[index_p + 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 2])*(p[index_p + N + 1]- p[index_p + N])/hx)
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2 * (l_w(Sw[index_Sw_xP    ])*(p[index_p - 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 1])*(p[index_p + N - 1]- p[index_p + N])/hx)
                else:
                    boundary1 = hy/2 * (l_w(Sw[index_Sw_xP + 1])*(p[index_p + 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 2])*(p[index_p + N + 1]- p[index_p + N])/hx)
                    boundary2 = hy/2 * (l_w(Sw[index_Sw_xP    ])*(p[index_p - 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 1])*(p[index_p + N - 1]- p[index_p + N])/hx)

            ## vertical flow
            if j == 0:
                boundary4l = u_inj*hx
                dpress3    = (p[index_p + 2 * N]-p[index_p + N] - hy*u_inj/l_w(Sw[index_Swt]))/2/hy
                if dpress3 < 0:
                    boundary3l = hx * l_w(Sw[index_Swt]) * dpress3
                else:
                    boundary3l = hx * l_w(Sw[index_Swt + N]) * dpress3
                boundary3h = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p + N] - hy*u_inj/l_w(Sw[index_Swt]))/2/hy
                boundary4h = u_inj*hx
            elif j == 1:
                dpress3    = (p[index_p + 2 * N]-p[index_p    ])/2/hy
                dpress4    = (p[index_p        ]+ hy*u_inj/l_w(Sw[index_Swt-N])-p[index_p + N])/2/hy
                if dpress3 < 0:
                    boundary3l = hx * l_w(Sw[index_Swt]) * dpress3
                else:
                    boundary3l = hx * l_w(Sw[index_Swt + N]) * dpress3
                if dpress4 > 0:
                    boundary4l = hx * l_w(Sw[index_Swt - N]) * dpress4
                else:
                    boundary4l = hx * l_w(Sw[index_Swt]) * dpress4
                boundary3h = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
                boundary4h = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p        ]+ hy*u_inj/l_w(Sw[index_Swt-N])-p[index_p + N])/2/hy
            elif j == M:
                boundary3l = 0
                dpress4    = (p[index_p - N    ]-p[index_p    ])/2/hy
                if dpress4 > 0:
                    boundary4l = hx * l_w(Sw[index_Swt - N]) * dpress4
                else:
                    boundary4l = hx * l_w(Sw[index_Swt]) * dpress4
                boundary3h = 0
                boundary4h = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p    ])/2/hy
            elif j == M-1:
                dpress3    = (p[index_p + 1 * N]-p[index_p    ])/2/hy
                dpress4    = (p[index_p - N    ]-p[index_p + N])/2/hy
                if dpress3 < 0:
                    boundary3l = hx * l_w(Sw[index_Swt]) * dpress3
                else:
                    boundary3l = hx * l_w(Sw[index_Swt + N]) * dpress3
                if dpress4 > 0:
                    boundary4l = hx * l_w(Sw[index_Swt - N]) * dpress4
                else:
                    boundary4l = hx * l_w(Sw[index_Swt]) * dpress4
                boundary3h = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 1 * N]-p[index_p    ])/2/hy
                boundary4h = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p + N])/2/hy
            else:
                dpress3    = (p[index_p + 2 * N]-p[index_p    ])/2/hy
                dpress4    = (p[index_p - N    ]-p[index_p + N])/2/hy
                if dpress3 < 0:
                    boundary3l = hx * l_w(Sw[index_Swt]) * dpress3
                else:
                    boundary3l = hx * l_w(Sw[index_Swt + N]) * dpress3
                if dpress4 > 0:
                    boundary4l = hx * l_w(Sw[index_Swt - N]) * dpress4
                else:
                    boundary4l = hx * l_w(Sw[index_Swt]) * dpress4
                boundary3h = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
                boundary4h = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - N    ]-p[index_p + N])/2/hy

            NumElementsSw = np.size(Sw)

            # u1 = (Sw[index_Sw_yP] + Sw[index_Sw_yP + N]) / 2
            # r1 = NanDiff(u1 - Sw[index_Swt - 1], Sw[index_Swt] - u1)
            # boundary1 = boundary1l + limiter(r1) * (boundary1h - boundary1l)
            #
            # u2 = (Sw[index_Sw_yP + 1] + Sw[index_Sw_yP + N + 1]) / 2
            # r2 = NanDiff(u2 - Sw[index_Swt], Sw[(index_Swt + 1) % NumElementsSw] - u2)
            # boundary2 = boundary2l + limiter(r2) * (boundary2h - boundary2l)

            u3 = (Sw[(index_Sw_xP + N + 1) % NumElementsSw] + Sw[(index_Sw_xP + N + 2) % NumElementsSw]) / 2
            r3 = NanDiff(u3 - Sw[index_Swt], Sw[(index_Swt + N) % NumElementsSw] - u3)
            boundary3 = boundary3l + limiter(r3) * (boundary3h - boundary3l)

            u4 = (Sw[index_Sw_xP] + Sw[(index_Sw_xP + 1) % NumElementsSw]) / 2
            r4 = NanDiff(u4 - Sw[index_Swt - N], Sw[index_Swt] - u4)
            boundary4 = boundary4l + limiter(r4) * (boundary4h - boundary4l)

            Swt[index_Swt] = (boundary1 + boundary2 + boundary3 + boundary4)/phi/hx/hy
            if j == 0 or j == M:
                Swt[index_Swt] = Swt[index_Swt]*2
    return Swt

# @jit(nopython=True)
def FVM_Sw_t_3(p,Sw,N,M,hx,hy):
    Swt = np.zeros(N*(M+1) + (N+1)*M)

    #################################################################### N + 1 x M

    ## vertical flows

    # vertices
    i = 0; j = 0
    index_Swt=N*(M+1)+i+(N+1)*j
    index_p=i-1+j*N
    index_Sw_yP=i-1+j*N

    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p + 1 + N] - p[index_p + 1])/hy
    boundary4 = u_inj*hx/2
    Swt[index_Swt]+=(boundary3+boundary4)/phi/hx/hy * 2

    i = 0; j = M - 1
    index_Swt=N*(M+1)+i+(N+1)*j
    index_p=i-1+j*N
    index_Sw_yP=i-1+j*N

    boundary3 = 0
    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ]) * (p[index_p + 1 - N] - p[index_p + 1])/hy
    Swt[index_Swt]+=(boundary3+boundary4)/phi/hx/hy * 2


    i = N; j = 0
    index_Swt=N*(M+1)+i+(N+1)*j
    index_p=i-1+j*N
    index_Sw_yP=i-1+j*N

    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N]) * (p[index_p + N] - p[index_p])/hy
    boundary4 = u_inj*hx/2
    Swt[index_Swt]+=(boundary3+boundary4)/phi/hx/hy * 2

    i = N; j = M - 1
    index_Swt=N*(M+1)+i+(N+1)*j
    index_p=i-1+j*N
    index_Sw_yP=i-1+j*N

    boundary3 = 0
    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ]) * (p[index_p - N] - p[index_p])/hy
    Swt[index_Swt] += (boundary3+boundary4)/phi/hx/hy * 2


    # i = 0 , i = N+1
    for j in range(1,M-1):
        i = 0
        index_Swt=N*(M+1)+i+(N+1)*j
        index_p=i-1+j*N
        index_Sw_yP=i-1+j*N

        boundary3=hx/2*l_w(Sw[index_Sw_yP+N+1])*(p[index_p+1+N]-p[index_p+1])/hy
        boundary4=hx/2*l_w(Sw[index_Sw_yP+1])*(p[index_p+1-N]-p[index_p+1])/hy
        Swt[index_Swt] += (boundary3 + boundary4) / phi / hx / hy * 2

        i = N
        index_Swt=N*(M+1)+i+(N+1)*j
        index_p=i-1+j*N
        index_Sw_yP=i-1+j*N

        boundary3=hx/2*l_w(Sw[index_Sw_yP+N])*(p[index_p+N]-p[index_p])/hy
        boundary4=hx/2*l_w(Sw[index_Sw_yP])*(p[index_p-N]-p[index_p])/hy
        Swt[index_Swt] += (boundary3 + boundary4) / phi / hx / hy * 2

    # j = 0, j = M
    for i in range(1,N):
        j = 0
        index_Swt = N * (M + 1) + i + (N + 1) * j
        index_p = i - 1 + j * N
        index_Sw_yP = i - 1 + j * N

        boundary3 = hx/2 * ( l_w(Sw[index_Sw_yP + N])* (p[index_p+N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p+N+1] - p[index_p + 1])/hy)
        boundary4 = u_inj*hx
        Swt[index_Swt]+=(boundary3+boundary4)/phi/hx/hy


        j = M-1
        index_Swt = N * (M + 1) + i + (N + 1) * j
        index_p = i - 1 + j * N
        index_Sw_yP = i - 1 + j * N

        boundary3 = 0
        boundary4 = hx/2 * ( l_w(Sw[index_Sw_yP + 0])* (p[index_p-N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + 0 + 1]) * (p[index_p-N+1] - p[index_p + 1])/hy)
        Swt[index_Swt] += (boundary3 + boundary4) / phi / hx / hy

    for i in range(1,N):
        for j in range(1,M-1):
            index_Swt   = N * (M + 1) + i + (N + 1) * j
            index_p     = i - 1 + j * N
            index_Sw_yP = i - 1 + j * N

            boundary3 = hx/2 * ( l_w(Sw[index_Sw_yP + N])* (p[index_p+N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + N + 1]) * (p[index_p+N+1] - p[index_p + 1])/hy)
            boundary4 = hx/2 * ( l_w(Sw[index_Sw_yP + 0])* (p[index_p-N]-p[index_p])/hy + l_w(Sw[index_Sw_yP + 0 + 1]) * (p[index_p-N+1] - p[index_p + 1])/hy)
            Swt[index_Swt] += (boundary3 + boundary4)/phi/hx/hy


    # horizontal flows
    for j in range(1,M):
        i = 0
        index_Swt = N * (M + 1) + i + (N + 1) * j
        index_p = i - 1 + j * N
        index_Sw_yP = i - 1 + j * N

        boundary1 = 0
        boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p + 1]) / 2 / hx
        Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy * 2

        i = 1
        index_Swt = N * (M + 1) + i + (N + 1) * j
        index_p = i - 1 + j * N
        index_Sw_yP = i - 1 + j * N

        boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p    ] - p[index_p + 1]) / 2 / hx
        boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p    ]) / 2 / hx
        Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy

        i = N - 1
        index_Swt = N * (M + 1) + i + (N + 1) * j
        index_p = i - 1 + j * N
        index_Sw_yP = i - 1 + j * N

        boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p + 1]) / 2 / hx
        boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 1] - p[index_p    ]) / 2 / hx
        Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy

        i = N
        index_Swt = N * (M + 1) + i + (N + 1) * j
        index_p = i - 1 + j * N
        index_Sw_yP = i - 1 + j * N

        boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p    ]) / 2 / hx
        boundary2 = 0
        Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy * 2

    for j in range(M):
        for i in range(2,N-1):
            index_Swt   = N * (M + 1) + i + (N + 1) * j
            index_p     = i - 1 + j * N
            index_Sw_yP = i - 1 + j * N

            boundary1 = hy * (l_w(Sw[index_Sw_yP    ]) + l_w(Sw[index_Sw_yP + N    ])) / 2 * (p[index_p - 1] - p[index_p + 1]) / 2 / hx
            boundary2 = hy * (l_w(Sw[index_Sw_yP + 1]) + l_w(Sw[index_Sw_yP + N + 1])) / 2 * (p[index_p + 2] - p[index_p    ]) / 2 / hx
            Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy


    ############################################################################################### N x M + 1
    # horizontal flow:
    # i = 0; j = 0
    # index_Swt = i + N * j
    # index_p = i + (j - 1) * N
    # index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)
    #
    # boundary1 = hy / 2 * l_w(Sw[index_Sw_xP + 1 + N + 1]) * (p[index_p + N + 1] - p[index_p + N]) / hx
    # boundary2 = 0
    # Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy * 2

    i = 0; j = M
    index_Swt = i + N * j
    index_p = i + (j - 1) * N
    index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

    boundary1 = hy/2*l_w(Sw[index_Sw_xP+1])*(p[index_p+1]-p[index_p])/hx
    boundary2 = 0
    Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy * 2

    # i = N-1; j = 0
    # index_Swt = i + N * j
    # index_p = i + (j - 1) * N
    # index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)
    #
    # boundary1 = 0
    # boundary2 = hy/2*l_w(Sw[index_Sw_xP + 0 + N + 1]) * (p[index_p + N - 1] - p[index_p + N])/hx
    # Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy * 2

    i = N-1; j = M
    index_Swt = i + N * j
    index_p = i + (j - 1) * N
    index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

    boundary1 = 0
    boundary2 = hy / 2 * l_w(Sw[index_Sw_xP + 0]) * (p[index_p - 1] - p[index_p]) / hx
    Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy * 2

    for j in range(1,M):
        i = 0
        index_Swt = i + N * j
        index_p = i + (j - 1) * N
        index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

        boundary1 = hy/2 * (l_w(Sw[index_Sw_xP + 1])*(p[index_p + 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 2])*(p[index_p + N + 1]- p[index_p + N])/hx)
        boundary2 = 0
        Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy

        i = N - 1
        index_Swt = i + N * j
        index_p = i + (j - 1) * N
        index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

        boundary1 = 0
        boundary2 = hy/2 * (l_w(Sw[index_Sw_xP    ])*(p[index_p - 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 1])*(p[index_p + N - 1]- p[index_p + N])/hx)
        Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy

    for i in range(1,N-1):
        # j = 0
        # index_Swt = i + N * j
        # index_p = i + (j - 1) * N
        # index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)
        #
        # boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1]) * (p[index_p + N + 1] - p[index_p + N])/hx
        # boundary2 = hy/2*l_w(Sw[index_Sw_xP + 0 + N + 1]) * (p[index_p + N - 1] - p[index_p + N])/hx
        # Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy * 2

        j = M
        index_Swt = i + N * j
        index_p = i + (j - 1) * N
        index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

        boundary1 = hy / 2 * l_w(Sw[index_Sw_xP + 1]) * (p[index_p + 1] - p[index_p]) / hx
        boundary2 = hy / 2 * l_w(Sw[index_Sw_xP + 0]) * (p[index_p - 1] - p[index_p]) / hx
        Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy * 2

    for i in range(1,N-1):
        for j in range(1,M):
            index_Swt = i + N * j
            index_p = i + (j - 1) * N
            index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

            boundary1 = hy/2 * (l_w(Sw[index_Sw_xP + 1])*(p[index_p + 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 2])*(p[index_p + N + 1]- p[index_p + N])/hx)
            boundary2 = hy/2 * (l_w(Sw[index_Sw_xP    ])*(p[index_p - 1]-p[index_p])/hx + l_w(Sw[index_Sw_xP + N + 1])*(p[index_p + N - 1]- p[index_p + N])/hx)
            Swt[index_Swt] += (boundary1 + boundary2) / phi / hx / hy

    for i in range(N):
        # j = 0
        # index_Swt = i + N * j
        # index_p = i + (j - 1) * N
        # index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)
        #
        # boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p + N] - hy*u_inj/l_w(Sw[index_Swt]))/2/hy
        # boundary4 = u_inj*hx
        # Swt[index_Swt] += (boundary3 + boundary4) / phi / hx / hy * 2

        j = 1
        index_Swt = i + N * j
        index_p = i + (j - 1) * N
        index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

        boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
        boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p        ]+ hy*u_inj/l_w(Sw[index_Swt-1*N])-p[index_p + N])/2/hy
        Swt[index_Swt] += (boundary3 + boundary4) / phi / hx / hy

        j = M-1
        index_Swt = i + N * j
        index_p = i + (j - 1) * N
        index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

        boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 1 * N]-p[index_p    ])/2/hy
        boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - 1*N    ]-p[index_p + N])/2/hy
        Swt[index_Swt] += (boundary3 + boundary4) / phi / hx / hy

        j = M
        index_Swt = i + N * j
        index_p = i + (j - 1) * N
        index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

        boundary3 = 0
        boundary4 = hx * (l_w(Sw[index_Sw_xP]) + l_w(Sw[index_Sw_xP + 1])) / 2 * (p[index_p - N] - p[index_p]) / 2 / hy
        Swt[index_Swt] += (boundary3 + boundary4) / phi / hx / hy * 2

    for j in range(2,M-1):
        for i in range(N):
            index_Swt = i + N * j
            index_p = i + (j - 1) * N
            index_Sw_xP = N * (M + 1) + i + (j - 1) * (N + 1)

            boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1])+l_w(Sw[index_Sw_xP + N + 2]))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
            boundary4 = hx*(l_w(Sw[index_Sw_xP        ])+l_w(Sw[index_Sw_xP     + 1]))/2*(p[index_p - 1*N    ]-p[index_p + N])/2/hy
            Swt[index_Swt] += (boundary3 + boundary4) / phi / hx / hy

    return Swt

def FVM_plot_Sw(Sw,N,M,L,W,c):
    Sw_plot=np.zeros([2*M+1,2*N+1])
    Sw1=Sw[0:(N*(M+1))].reshape(M+1,N)

    Sw2=Sw[(N*(M+1)):].reshape(M,N+1)

    for i in range(N):
        for j in range(M+1):
            index_i=2*i+1
            index_j=2*j
            Sw_plot[index_j,index_i]=Sw1[j,i]

    for i in range(N+1):
        for j in range(M):
            index_i=2*i
            index_j=2*j+1
            Sw_plot[index_j,index_i]=Sw2[j,i]

    for i in range(2*N+1):
        for j in range(2*M+1):
            if (i+j)%2==0:
                b1=0
                b2=0
                b3=0
                b4=0
                count=0

                if j == 0:
                    b2 = 0
                elif j == 2*M:
                    b1 = Sw_plot[j-1,i]
                    count = count + 1
                else:
                    b1=Sw_plot[j-1,i]
                    b2=Sw_plot[j+1,i]
                    count=count+2

                if i==0:
                    b4=Sw_plot[j,i+1]
                    count=count+1
                elif i==2*N:
                    b3=Sw_plot[j,i-1]
                    count=count+1
                else:
                    b4=Sw_plot[j,i+1]
                    b3=Sw_plot[j,i-1]
                    count=count+2

                Sw_plot[j,i]=(b1+b2+b3+b4)/count

    return Sw_plot

def FVM_plot_Sw_Side(Sw,N,M,L,W,c):
    Sw_plot=np.zeros([2*M+1,2*N+1])
    Sw1=Sw[0:(N*(M+1))].reshape(M+1,N)

    Sw2=Sw[(N*(M+1)):].reshape(M,N+1)

    for i in range(N):
        for j in range(M+1):
            index_i=2*i+1
            index_j=2*j
            Sw_plot[index_j,index_i]=Sw1[j,i]

    for i in range(N+1):
        for j in range(M):
            index_i=2*i
            index_j=2*j+1
            Sw_plot[index_j,index_i]=Sw2[j,i]

    for i in range(2*N+1):
        for j in range(2*M+1):
            if (i+j)%2==0:
                b1=0
                b2=0
                b3=0
                b4=0
                count=0

                if j == 0:
                    b2 = 0
                # elif j == 2*M:
                    # b1 = Sw_plot[j-1,i]
                    # count = count + 1
                # else:

                    # b1=Sw_plot[j-1,i]
                    # b2=Sw_plot[j+1,i]
                    # count=count+2

                if i==0:
                    b4=Sw_plot[j,i+1]
                    count=count+1
                elif i==2*N:
                    b3=Sw_plot[j,i-1]
                    count=count+1
                else:
                    b4=Sw_plot[j,i+1]
                    b3=Sw_plot[j,i-1]
                    count=count+2

                Sw_plot[j,i]=(b1+b2+b3+b4)/count

    return Sw_plot

def FVM_diffusion(SwInput,N,M):
    Swdiff = np.copy(SwInput)

    Sw1=SwInput[0:(N*(M+1))].reshape(M+1,N)

    Sw2=SwInput[(N*(M+1)):].reshape(M,N+1)

    for i in range(N):
        for j in range(M+1):
            indexSwDiff = i + j * N
            indexSw     = N*(M+1) + i + (j-1)*(N+1)

            if j == 0:
                continue
                Swdiff[indexSwDiff] = (SwInput[indexSw + N + 1] + SwInput[indexSw + N + 2]) / 2
            elif j == M:
                continue
                Swdiff[indexSwDiff] = (SwInput[indexSw        ] + SwInput[indexSw     + 1]) / 2
            else:
                Swdiff[indexSwDiff] = (SwInput[indexSw + N + 1] + SwInput[indexSw + N + 2] + SwInput[indexSw] + SwInput[indexSw + 1]) / 4

    for i in range(N+1):
        for j in range(M):
            indexSwDiff = N*(M+1) + i + j * (N+1)
            indexSw     = i - 1 + j * N

            if i == 0:
                Swdiff[indexSwDiff] = (SwInput[indexSw + 1] + SwInput[indexSw + N + 1]) / 2
            elif i == N:
                Swdiff[indexSwDiff] = (SwInput[indexSw    ] + SwInput[indexSw + N    ]) / 2
            else:
                Swdiff[indexSwDiff] = (SwInput[indexSw + 1] + SwInput[indexSw + N + 1] + SwInput[indexSw] + SwInput[indexSw + N]) / 4

    return Swdiff




# print(p.reshape(M,N))

# import plotly.graph_objects as go
# fig = go.Figure(data=[go.Surface( z=p.reshape(M,N),x = np.linspace(0,W,N), y = np.linspace(0,L,M))])
# fig.show()

#
Sw_plot = FVM_plot_Sw(Sw,N,M,L,W,c)
p       = FVM_pressure(Sw,N,M,hx,hy)
Swt     = FVM_Sw_t_3(p,Sw,N,M,hx,hy)
# for i in range(2):
#     Sw += dt*FVM_Sw_t_4(p,Sw,N,M,hx,hy)
#     Swt     = dt*FVM_Sw_t_4(p,Sw,N,M,hx,hy)
# print(Swt)


x = np.linspace(0,W,2*N+1)
y = np.linspace(0,L,2*M+1)
X,Y = np.meshgrid(x, y)

plt.ion()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Sw_plot,cmap="coolwarm",linewidth=0,antialiased=False)

ax.view_init(20, 30)
fig.canvas.draw()
contour_axis = plt.gca()

jdx = 4
for i in range(N):
    Sw[N*jdx+i] += delta*(1+math.sin(kNum*2*math.pi*i*hx/W))



## run newton backward.
# for a speedup one can calculate the pressure once outside the while loop. This seems to give similar solutions. I am however not sure if it converges correctly...
for i in tqdm.tqdm(range(Tstep)):


    Sw0 = Sw

    error = 1
    # p = FVM_pressure(Sw, N, M, hx, hy)
    # Swt     = FVM_Sw_t_4(p,Sw,N,M,hx,hy)
    # Sw = Sw + dt*Swt

    # p       = FVM_pressure(Sw,N,M,hx,hy)
    while error > eps:
        # p       = FVM_pressure_CG(p, Sw, N, M, hx, hy)
        p = FVM_pressure(Sw, N, M, hx, hy)
        Swt     = FVM_Sw_t_3(p,Sw,N,M,hx,hy)
        Sw_new  = Sw0 + dt * Swt

        error   = np.linalg.norm(np.divide(Sw_new-Sw,Sw0),np.inf)
        Sw      = Sw_new
        # print(error)

    Sw_plot = FVM_plot_Sw(Sw, N, M, L, W, c)
    ax = plt.axes(projection='3d')
    ax.view_init(20, 0+60*i/Tstep)
    ax.plot_surface(X, Y, Sw_plot,cmap="coolwarm",linewidth=0,antialiased=True)

    fig.canvas.draw()
    contour_axis = plt.gca()



## plotting
if plotting:
    Sw_plot = FVM_plot_Sw(Sw,N,M,L,W,c)
    import plotly.graph_objects as go

    fig=go.Figure(data=[go.Surface(z=Sw_plot,x=np.linspace(0,W,2*N+1),y=np.linspace(0,L,2*M+1))])
    fig.show()

# import plotly.graph_objects as go
# fig = go.Figure(data=[go.Surface( z=p.reshape(M,N),x = np.linspace(0,W,N), y = np.linspace(0,L,M))])
# fig.show()
#
# timeP_Mat = timeit(
#  lambda: FVM_pressure_Mat(Sw,N,M,hx,hy),
#  number=100
# )
# print("Pressure Mat Calc:",timeP_Mat/100)
#
#
#
# timeP = timeit(
#  lambda: FVM_pressure(Sw,N,M,hx,hy),
#  number=100
# )
# print("Pressure Calc:",(timeP-timeP_Mat)/100)
#
# p = FVM_pressure(Sw,N,M,hx,hy)
#
#
# timePcg = timeit(
#  lambda: FVM_pressure_CG(p,Sw,N,M,hx,hy),
#  number=1
# )
# print("Pressure CG Calc:",timePcg/1)
#
# timeSwt = timeit(
#  lambda: FVM_Sw_t_3(p,Sw,N,M,hx,hy),
#  number=100
# )
# print("Swt Calc:",timeSwt/100)


# Swt1 = Swt[0:( N * (M + 1))]
# Swt2 = Swt[( N * (M + 1)):]
#
# Sw1 = Sw[0:( N * (M + 1))]
# Sw2 = Sw[( N * (M + 1)):]

# import plotly.graph_objects as go
# fig = go.Figure(data=[go.Surface( z=Swt.reshape(M,N+1),x = np.linspace(0,W,N+1), y = np.linspace(0,L,M))])
# fig.show()
# print("N = ",N)
# print("M = ",M)
#
# print("pressure")
# print(p.reshape(M,N))
#
# print("Sw")
# print(Sw1.reshape(M+1,N))
# print(Sw2.reshape(M,N+1))
#
# print("Swt")
# print(Swt1.reshape(M+1,N))
# print(Swt2.reshape(M,N+1))
#









