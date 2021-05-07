import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
from reservoirModule import *
import scipy.sparse
from timeit import timeit



hx = 1
hy = .5
W  = 5
L  = 6


t_end = 0.2

eps = 1e-1

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

N = int(W / hx)
M = int(L / hy)


def magic_function(x, c):
    return df_dSw(x, c) - (f_w(x, c) - f_w(c.S_wc, c))/(x - c.S_wc)

S_w_shock = bisection(magic_function, (c.S_wc, 1 - c.S_or), 100, c)
shockspeed = c.u_inj/c.phi*df_dSw(S_w_shock, c)

# why the constant 5?
dt = min(hx,hy)/shockspeed/2/5
Tstep = int(t_end/dt)

print("shockspeed = ",shockspeed,", shock position = ",shockspeed*dt*Tstep,", maximum dt = ",min(hx,hy)/shockspeed/2)


Sw = 0.1*np.ones(N*(M+1) + (N+1)*M)
for i in range(N):
    Sw[i] = 0.9


def FVM_pressure(Sw,c,N,M,hx,hy):
    p  = np.zeros(N*M)


    # set up S matrix for each volume, and each pressure
    S = np.zeros( [N*M, p.size])

    for i in range(N):
        for j in range(M):
            index_p = i + j * N
            index_Sw_xP = N * (M + 1) + i + j * (N + 1)
            index_Sw_xN = N * (M + 1) + i + 1 + j * (N + 1)
            index_Sw_yN = i + (j + 1) * N
            index_Sw_yP = i + j * N


            if i==0:
                if j == 0:
                        S[index_p, index_p]     = -hy / hx * l_t(Sw[index_Sw_xN], c) - hx / hy * l_t(Sw[index_Sw_yN], c)
                        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN], c)
                        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN], c)
                elif j == M-1:
                        S[index_p, index_p]     = -hy / hx * l_t(Sw[index_Sw_xN], c) - hx / hy * l_t(Sw[index_Sw_yP], c)
                        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN], c)
                        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP], c)
                else:
                        S[index_p, index_p]     = -hy / hx * l_t(Sw[index_Sw_xN], c) - hx / hy * l_t(Sw[index_Sw_yN], c) - hx / hy * l_t(Sw[index_Sw_yP], c)
                        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN], c)
                        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN], c)
                        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP], c)
            elif i == N-1:
                if j == 0:
                        S[index_p, index_p]     = - hy / hx * l_t(Sw[index_Sw_xP],c) - hx / hy * l_t(Sw[index_Sw_yN], c)
                        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP], c)
                        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN], c)
                elif j == M-1:
                        S[index_p, index_p]     = - hy / hx * l_t(Sw[index_Sw_xP],c) - hx / hy * l_t(Sw[index_Sw_yP], c)
                        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP], c)
                        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP], c)
                else:
                        S[index_p, index_p]     = - hy / hx * l_t(Sw[index_Sw_xP],c) - hx / hy * l_t(Sw[index_Sw_yN], c) - hx / hy * l_t(Sw[index_Sw_yP], c)
                        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP], c)
                        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN], c)
                        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP], c)
            else:
                if j == 0:
                        S[index_p, index_p]     = -hy / hx * l_t(Sw[index_Sw_xN], c)  - hy / hx * l_t(Sw[index_Sw_xP],c) - hx / hy * l_t(Sw[index_Sw_yN], c)
                        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN], c)
                        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP], c)
                        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN], c)
                elif j == M-1:
                        S[index_p, index_p]     = -hy / hx * l_t(Sw[index_Sw_xN], c)  - hy / hx * l_t(Sw[index_Sw_xP],c) - hx / hy * l_t(Sw[index_Sw_yP], c)
                        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN], c)
                        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP], c)
                        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP], c)
                else:
                        S[index_p, index_p]     = -hy / hx * l_t(Sw[index_Sw_xN], c)  - hy / hx * l_t(Sw[index_Sw_xP],c) - hx / hy * l_t(Sw[index_Sw_yN], c) - hx / hy * l_t(Sw[index_Sw_yP], c)
                        S[index_p, index_p + 1] = hy / hx * l_t(Sw[index_Sw_xN], c)
                        S[index_p, index_p - 1] = hy / hx * l_t(Sw[index_Sw_xP], c)
                        S[index_p, index_p + N] = hx / hy * l_t(Sw[index_Sw_yN], c)
                        S[index_p, index_p - N] = hx / hy * l_t(Sw[index_Sw_yP], c)

    f = np.zeros(p.size+1)
    for i in range(N):
        for j in range(M):
            index_p = i + j * N
            if j == 0:
                f[index_p] = -hx*c.u_inj
            elif j == M-1:
                f[index_p] = hx*c.u_inj

    l = np.zeros(p.size)
    l[-1] = 1
    S = np.vstack([S,l])

    p = np.linalg.lstsq(S,f,rcond=None)
    return p[0]

def FVM_Sw_t(p,Sw,c,N,M,hx,hy):
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
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1],c) * (p[index_p + 1 + N] - p[index_p + 1])/hy
                    boundary4 = c.u_inj*hx/2
                elif j == 1:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1],c) * (p[index_p + 1 + N] - p[index_p + 1])/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ],c) * (p[index_p + 1 - N] - p[index_p + 1])/hy
                elif j == M-1:
                    boundary3 = 0
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ],c) * (p[index_p + 1 - N] - p[index_p + 1])/hy
                else:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N + 1],c) * (p[index_p + 1 + N    ] - p[index_p + 1 - N])/2/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP + 1    ],c) * (p[index_p + 1 - 2 * N] - p[index_p + 1    ])/2/hy
            elif i == N:
                if j == 0:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N],c) * (p[index_p     + N] - p[index_p    ])/hy
                    boundary4 = c.u_inj*hx/2
                elif j == 1:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N],c) * (p[index_p     + N] - p[index_p    ])/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ],c) * (p[index_p     - N] - p[index_p    ])/hy
                elif j == M-1:
                    boundary3 = 0
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ],c) * (p[index_p     - N] - p[index_p    ])/hy
                else:
                    boundary3 = hx/2*l_w(Sw[index_Sw_yP + N],c) * (p[index_p     + N] - p[index_p - N])/2/hy
                    boundary4 = hx/2*l_w(Sw[index_Sw_yP    ],c) * (p[index_p - 2 * N] - p[index_p    ])/2/hy
            else:
                if j==0:
                    boundary3 = hx*(l_w(Sw[index_Sw_yP + N],c)+l_w(Sw[index_Sw_yP + N + 1],c))/2 * (p[index_p+N] + p[index_p+1+N] - p[index_p] - p[index_p+1])/2/hy
                    boundary4 = c.u_inj*hx
                elif j == 1:
                    boundary3 = hx*(l_w(Sw[index_Sw_yP + N],c)+l_w(Sw[index_Sw_yP + N + 1],c))/2 * (p[index_p+N] + p[index_p+1+N] - p[index_p] - p[index_p+1])/2/hy
                    boundary4 = hx*(l_w(Sw[index_Sw_yP    ],c)+l_w(Sw[index_Sw_yP + 1    ],c))/2 * (p[index_p-N] + p[index_p+1-N] - p[index_p] - p[index_p+1])/2/hy
                elif j==M-1:
                    boundary3 = 0
                    boundary4 = hx*(l_w(Sw[index_Sw_yP    ],c)+l_w(Sw[index_Sw_yP + 1    ],c))/2 * (p[index_p-N] + p[index_p+1-N] - p[index_p] - p[index_p+1])/2/hy
                else:
                    boundary3 = hx*(l_w(Sw[index_Sw_yP + N],c)+l_w(Sw[index_Sw_yP + N + 1],c))/2 * (p[index_p + N] + p[index_p+1+N  ] - p[index_p-N] - p[index_p+1-N])/4/hy
                    boundary4 = hx*(l_w(Sw[index_Sw_yP    ],c)+l_w(Sw[index_Sw_yP + 1    ],c))/2 * (p[index_p-2*N] + p[index_p+1-2*N] - p[index_p  ] - p[index_p+1  ])/4/hy

            ## horizonatal flows
            if i==0:
                boundary1 = 0
                boundary2 = hy * (l_w(Sw[index_Sw_yP + 1], c) + l_w(Sw[index_Sw_yP + N + 1], c)) / 2 * (p[index_p + 2] - p[index_p + 1]) / hx
            elif i==N:
                boundary1 = hy * (l_w(Sw[index_Sw_yP    ], c) + l_w(Sw[index_Sw_yP + N    ], c)) / 2 * (p[index_p - 1] - p[index_p    ]) / hx
                boundary2 = 0
            else:
                boundary1 = hy * (l_w(Sw[index_Sw_yP    ], c) + l_w(Sw[index_Sw_yP + N    ], c)) / 2 * (p[index_p    ] - p[index_p + 1]) / hx
                boundary2 = hy * (l_w(Sw[index_Sw_yP + 1], c) + l_w(Sw[index_Sw_yP + N + 1], c)) / 2 * (p[index_p + 1] - p[index_p    ]) / hx

            Swt[index_Swt] = (boundary1 + boundary2 + boundary3 + boundary4)/c.phi/hx/hy
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
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1],c) * (2*p[index_p+1+N]+hx*c.u_inj/l_w(Sw[index_Swt+1],c)-2*p[index_p+N]-hx*c.u_inj/l_w(Sw[index_Swt],c))/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP     + N + 1],c) * (2*p[index_p+N]+hx*c.u_inj/l_w(Sw[index_Swt],c)-2*p[index_p - 1 + N]-hx*c.u_inj/l_w(Sw[index_Swt-1],c))/hx
                else:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1 + N + 1],c) * (2*p[index_p+1+N]+hx*c.u_inj/l_w(Sw[index_Swt+1],c)-2*p[index_p+N]-hx*c.u_inj/l_w(Sw[index_Swt],c))/hx
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP     + N + 1],c) * (2*p[index_p+N]+hx*c.u_inj/l_w(Sw[index_Swt],c)-2*p[index_p - 1 + N]-hx*c.u_inj/l_w(Sw[index_Swt-1],c))/hx
            elif j == M:
                if i == 0:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP + 1    ],c) * (p[index_p + 1]-p[index_p])/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP + 1    ],c) * (p[index_p - 1]-p[index_p])/hx
                else:
                    boundary1 = hy/2*l_w(Sw[index_Sw_xP+1],c)*(p[index_p+1]-p[index_p])/hx
                    boundary2 = hy/2*l_w(Sw[index_Sw_xP+1],c)*(p[index_p-1]-p[index_p])/hx
            else:
                if i == 0:
                    boundary1 = hy*(l_w(Sw[index_Sw_xP + 1],c) + l_w(Sw[index_Sw_xP + 1 + N + 1],c))/2*(p[index_p + 1] + p[index_p + 1 + N] - p[index_p] - p[index_p + N])/2/hx
                    boundary2 = 0
                elif i == N-1:
                    boundary1 = 0
                    boundary2 = hy*(l_w(Sw[index_Sw_xP    ],c) + l_w(Sw[index_Sw_xP     + N + 1],c))/2*(p[index_p - 1] + p[index_p - 1 + N] - p[index_p] - p[index_p + N])/2/hx
                else:
                    boundary1 = hy*(l_w(Sw[index_Sw_xP + 1],c) + l_w(Sw[index_Sw_xP + 1 + N + 1],c))/2*(p[index_p + 1] + p[index_p + 1 + N] - p[index_p] - p[index_p + N])/2/hx
                    boundary2 = hy*(l_w(Sw[index_Sw_xP    ],c) + l_w(Sw[index_Sw_xP     + N + 1],c))/2*(p[index_p - 1] + p[index_p - 1 + N] - p[index_p] - p[index_p + N])/2/hx

            ## vertical flow
            if j == 0:
                continue
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1],c)+l_w(Sw[index_Sw_xP + N + 2],c))/2*(p[index_p + N    ]-p[index_p    ])/hy
                boundary4 = c.u_inj*hx
            if j == 1:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1],c)+l_w(Sw[index_Sw_xP + N + 2],c))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ],c)+l_w(Sw[index_Sw_xP     + 1],c))/2 * c.u_inj / l_w(Sw[index_Swt - N],c)   # estimate pressure difference with inflow pressure difference
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ],c)+l_w(Sw[index_Sw_xP     + 1],c))/2*(p[index_p] + hy * c.u_inj / l_w(Sw[index_Swt - N],c)  -p[index_p + N])/2/hy
            elif j == M:
                boundary3 = 0
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ],c)+l_w(Sw[index_Sw_xP     + 1],c))/2*(p[index_p - N    ]-p[index_p    ])/hy
            elif j == M-1:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1],c)+l_w(Sw[index_Sw_xP + N + 2],c))/2*(p[index_p + N    ]-p[index_p    ])/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ],c)+l_w(Sw[index_Sw_xP     + 1],c))/2*(p[index_p - N    ]-p[index_p + N])/2/hy
            else:
                boundary3 = hx*(l_w(Sw[index_Sw_xP + N + 1],c)+l_w(Sw[index_Sw_xP + N + 2],c))/2*(p[index_p + 2 * N]-p[index_p    ])/2/hy
                boundary4 = hx*(l_w(Sw[index_Sw_xP        ],c)+l_w(Sw[index_Sw_xP     + 1],c))/2*(p[index_p - N    ]-p[index_p + N])/2/hy

            Swt[index_Swt] = (boundary1 + boundary2 + boundary3 + boundary4)/c.phi/hx/hy
            if j == 0 or j == M:
                Swt[index_Swt] = Swt[index_Swt]*2
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

    import plotly.graph_objects as go
    fig=go.Figure(data=[go.Surface(z=Sw_plot,x=np.linspace(0,W,2*N+1),y=np.linspace(0,L,2*M+1))])
    fig.show()

    return

def FVM_diffusion_add(Sw,Swt,dt,N,M):
    Swdiff = Sw

    Sw1=Sw[0:(N*(M+1))].reshape(M+1,N)

    Sw2=Sw[(N*(M+1)):].reshape(M,N+1)

    for i in range(N):
        for j in range(M+1):
            indexSwDiff = i + j * N
            indexSw     = N*(M+1) + i + (j-1)*(N+1)

            if j == 0:
                continue
                Swdiff[indexSwDiff] = (Sw[indexSw + N + 1] + Sw[indexSw + N + 2]) / 2
            elif j == M:
                continue
                Swdiff[indexSwDiff] = (Sw[indexSw        ] + Sw[indexSw     + 1]) / 2
            else:
                Swdiff[indexSwDiff] = (Sw[indexSw + N + 1] + Sw[indexSw + N + 2] + Sw[indexSw] + Sw[indexSw + 1]) / 4

    for i in range(N+1):
        for j in range(M):
            indexSwDiff = N*(M+1) + i + j * (N+1)
            indexSw     = i - 1 + j * N

            if i == 0:
                Swdiff[indexSwDiff] = (Sw[indexSw + 1] + Sw[indexSw + N + 1]) / 2
            elif i == N:
                Swdiff[indexSwDiff] = (Sw[indexSw    ] + Sw[indexSw + N    ]) / 2
            else:
                Swdiff[indexSwDiff] = (Sw[indexSw + 1] + Sw[indexSw + N + 1] + Sw[indexSw] + Sw[indexSw + N]) / 4

    return Swdiff + dt * Swt




# print(p.reshape(M,N))

# import plotly.graph_objects as go
# fig = go.Figure(data=[go.Surface( z=p.reshape(M,N),x = np.linspace(0,W,N), y = np.linspace(0,L,M))])
# fig.show()




## run newton backward.
#for a speedup one can calculate the pressure once outside the while loop. This seems to give similar solutions. I am however not sure if it converges correctly...
for i in tqdm.tqdm(range(Tstep)):
    # if i == 94:
    #     idx = 10
    #     jdx = 9
    #     indexSw = N * (M + 1) + idx + jdx * (N + 1)
    #     Sw[indexSw] = 0.3
    Sw0 = Sw
    error = 1
    # p=FVM_pressure(Sw,c,N,M,hx,hy)
    while error > eps:
        p       = FVM_pressure(Sw,c,N,M,hx,hy)
        Swt     = FVM_Sw_t(p,Sw,c,N,M,hx,hy)
        # Sw_new  = FVM_diffusion_add(Sw0,Swt,dt,N,M)
        Sw_new  = Sw0 + dt * Swt
        error   = np.linalg.norm(np.divide(Sw_new-Sw,Sw0),np.inf)
        Sw      = Sw_new
        # print(error)
    # Sw = FVM_diffusion_add(Sw0,Swt,dt,N,M)

# Newton Forward, seems to only converge for small dt... This induces to much diffusion
# for i in tqdm.tqdm(range(Tstep)):
#     p = FVM_pressure(Sw, c, N, M, hx, hy)
#     Swt     = FVM_Sw_t(p,Sw,c,N,M,hx,hy)
#     Sw  = FVM_diffusion_add(Sw,Swt,dt,N,M)

# p       = FVM_pressure(Sw,c,N,M,hx,hy)
# Swt     = FVM_Sw_t(p,Sw,c,N,M,hx,hy)
# Sw      = FVM_diffusion_add(Sw,Swt,dt,N,M)

## plotting
FVM_plot_Sw(Sw,N,M,L,W,c)

# import plotly.graph_objects as go
# fig = go.Figure(data=[go.Surface( z=p.reshape(M,N),x = np.linspace(0,W,N), y = np.linspace(0,L,M))])
# fig.show()



# time = timeit(
#  lambda: FVM_pressure(Sw,c,N,M,hx,hy),
#  number=100
# )
# print("Pressure Calc:",time)
#
# p = FVM_pressure(Sw,c,N,M,hx,hy)
# time = timeit(
#  lambda: FVM_Sw_t(p,Sw,c,N,M,hx,hy),
#  number=100
# )
# print("Swt Calc:",time)


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









