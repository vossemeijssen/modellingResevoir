import numpy as np
import matplotlib.pyplot as plt
import tqdm
import math
from reservoirModule import *
from scipy.sparse import  diags

def calc_pressure(Sw,c):
    M,N = Sw.shape
    N = N + 2

    # add periodic BC to Sw
    Sw = np.hstack([np.expand_dims(Sw[:,-1], axis=1),Sw,np.expand_dims(Sw[:,0], axis=1)])
    Sw = Sw.reshape(M*N)

    # prealocate memory
    l_tot  = np.zeros(N*M)
    l_wL   = np.zeros(N*M)
    l_oL   = np.zeros(N*M)
    dl_tot = np.zeros(N*M)
    dl_totdx = np.zeros(N*M)
    dl_totdy = np.zeros(N*M)

    K = N*M
    for i in range(K):
        # calculate lambda_tot
        l_tot[i]   = l_t(Sw[i],c)
        l_wL[i]    = l_w(Sw[i],c)
        l_oL[i]    = l_o(Sw[i],c)
        dl_tot[i]  = dl_w(Sw[i],c)+dl_o(Sw[i],c)

    for i in range(K):
        # calculate position derivatives of l_tot,
        # since some BC are periodic they need to be calculated differently
        if i%N == N-1:
            dl_totdx[i] = 1 / 2 / c.dx * (l_tot[(i + 1-N)] - l_tot[(i - 1)])
        elif i%N == 0:
            dl_totdx[i] = 1 / 2 / c.dx * (l_tot[(i + 1)] - l_tot[(i - 1 + N)])
        else:
            dl_totdx[i] = 1 / 2 / c.dx * (l_tot[(i + 1)] - l_tot[(i - 1)])

        if math.floor(i/N) == M-1:
            dl_totdy[i] = 1 / c.dx * (3*l_tot[(i)] - 4 * l_tot[(i - N)] + l_tot[(i - 2 *N)])
        elif math.floor(i/N) == 0:
            # note that we have a less accurate calculation here
            dl_totdy[i] = 0*1 /2 / c.dx * (-l_tot[i+2*N]+ 4*l_tot[i+N] - 3* l_tot[(i)])
        else:
            dl_totdy[i] = 1 / 2 / c.dx * (l_tot[i + N] - l_tot[i-N])

        # set up diagonals
        diagonals = [-4 / c.dx / c.dx * l_tot,
                     l_tot[0:-1] / c.dx / c.dx + dl_totdx[0:-1] / 2 / c.dx,
                     l_tot[1:] / c.dx / c.dx - dl_totdx[1:] / 2 / c.dx,
                     l_tot[0:-N] / c.dx / c.dx + dl_totdy[0:-N] / 2 / c.dx,
                     l_tot[N:] / c.dx / c.dx - dl_totdy[N:] / 2 / c.dx]
        D = diags(diagonals, [0, 1, -1, N, -N]).toarray()



        # due to the periodic BC, we have to add the correct items in to the matrix (these are now in the
        # extra left and right elements and after we remove these, we have to add them back in.
        for j in range(M):
            index = j * (N) + (N - 2)
            D[index, j * N + 1] = (l_tot[index] / c.dx / c.dx + dl_totdx[index] / 2 / c.dx)
            index = j * (N) + 1
            D[index, j * N + N - 2] = (l_tot[index] / c.dx / c.dx - dl_totdx[index] / 2 / c.dx)

        # Since we have Neumann BC, we can eliminate the some of the values and need to add others:
        # we can get rid of the p_y terms, but we have to use a different stencil for the laplacian.
        for i in range(N - 1):
            index = i
            D[index, index + N] = 2 * (l_tot[index] / c.dx / c.dx)
            D[index, index + 1] = (l_tot[index] / c.dx / c.dx)
            D[index, index - 1] = (l_tot[index] / c.dx / c.dx)
            index = i + (M - 1) * (N)
            D[index, index - N] = 2 *(l_tot[index] / c.dx / c.dx)
            D[index, index + 1] = (l_tot[index] / c.dx / c.dx)
            D[index, index - 1] = (l_tot[index] / c.dx / c.dx)

    # create a list to remove the additional left and right elements (effectively we reduce N by 2)
    list = []
    for i in range(K):
        if i % N == 0:
            list.append(i)
        elif i % N == N - 1:
            list.append(i)
            
    # remove the additional columns and rows who belong to the additional left and right elements.
    D = np.delete(D,list,axis=1)
    D = np.delete(D,list,axis=0)

    # plt.matshow(D)
    # plt.show()

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
            l[i] = (0*dl_totdy[index] - 2/c.dx*l_tot[index])*c.u_inj/l_wL[index]
        if math.floor(i/(N-2)) == M-1:
            index = (M-1)*N + i % (N - 2) + 1
            l[i] = (dl_totdy[index] + 2/c.dx*l_tot[index])*c.u_inj/l_tot[index]

    # solve matrix vector product
    A = np.matmul(np.transpose(D),D)
    b = np.matmul(np.transpose(D),l)
    p = np.linalg.solve(A,b)

    residual = np.matmul(D,p)-l
    print(np.linalg.norm(residual)/(np.size(residual)))
    # plt.show()
    # print(np.sum(residual))
    return p

def calc_Swt(Sw,p,c):
    M, N = Sw.shape
    N = N + 2

    # add periodic BC to Sw
    Sw = np.hstack([np.expand_dims(Sw[:, -1], axis=1), Sw, np.expand_dims(Sw[:, 0], axis=1)])
    Sw = Sw.reshape(M * N)

    # prealocate memory
    l_tot = np.zeros(N * M)
    l_tot_tot = np.zeros(N*M)
    dl_tot = np.zeros(N * M)
    dl_totdx = np.zeros(N * M)
    dl_totdy = np.zeros(N * M)

    # for all elements, calculate the following
    K = N * M
    for i in range(K):
        # calculate lambda_tot
        l_tot[i] = l_w(Sw[i], c)
        l_tot_tot[i] = l_w(Sw[i], c)
        dl_tot[i] = dl_w(Sw[i], c)
    for i in range(K):
        # calculate position derivatives of l_tot,
        # since some BC are periodic they need to be calculated differently
        if i % N == N - 1:
            dl_totdx[i] = 1 / 2 / c.dx * (dl_tot[(i + 1 - N)] - dl_tot[(i - 1)])
        elif i % N == 0:
            dl_totdx[i] = 1 / 2 / c.dx * (dl_tot[(i + 1)] - dl_tot[(i - 1 + N)])
        else:
            dl_totdx[i] = 1 / 2 / c.dx * (dl_tot[(i + 1)] - dl_tot[(i - 1)])

        if math.floor(i/N) == M-1:
            dl_totdy[i] = 0*1 / c.dx * (3*dl_tot[(i)] - 4 * dl_tot[(i - N)] + dl_tot[(i - 2 *N)])
        elif math.floor(i/N) == 0:
            # note that we have a less accurate calculation here
            dl_totdy[i] = 0*1 /2 / c.dx * (-dl_tot[i+2*N]+ 4*dl_tot[i+N] - 3* dl_tot[(i)])
        else:
            dl_totdy[i] = 1 / 2 / c.dx * (dl_tot[i + N] - dl_tot[i-N])

        # set up diagonals
        diagonals = [-4 / c.dx / c.dx * l_tot,
                     l_tot[0:-1] / c.dx / c.dx + dl_totdx[0:-1] / 2 / c.dx,
                     l_tot[1:] / c.dx / c.dx - dl_totdx[1:] / 2 / c.dx,
                     l_tot[0:-N] / c.dx / c.dx + dl_totdy[0:-N] / 2 / c.dx,
                     l_tot[N:] / c.dx / c.dx - dl_totdy[N:] / 2 / c.dx]
        D = diags(diagonals, [0, 1, -1, N, -N]).toarray()

        # create a list to remove the additional left and right elements (effectively we reduce N by 2)
        list = []
        for i in range(K):
            if i % N == 0:
                list.append(i)
            elif i % N == N - 1:
                list.append(i)

        # due to the periodic BC, we have to add the correct items in to the matrix (these are now in the
        # extra left and right elements and after we remove these, we have to add them back in.
        for j in range(M):
            index = j * (N) + (N - 2)
            D[index, j * N + 1] = (l_tot[index] / c.dx / c.dx + dl_totdx[index] / 2 / c.dx)
            index = j * (N) + 1
            D[index, j * N + N - 2] = (l_tot[index] / c.dx / c.dx - dl_totdx[index] / 2 / c.dx)

        # Since we have Neumann BC, we can eliminate the some of the values and need to add others:
        # we can get rid of the p_y terms, but we have to use a different stencil for the laplacian.
        for i in range(N - 1):
            index = i
            D[index, index + N] = 2 * (l_tot[index] / c.dx / c.dx)
            D[index, index + 1] = (l_tot[index] / c.dx / c.dx)
            D[index, index - 1] = (l_tot[index] / c.dx / c.dx)
            index = i + (M - 1) * (N)
            D[index, index - N] = 2 * (l_tot[index] / c.dx / c.dx)
            D[index, index + 1] = (l_tot[index] / c.dx / c.dx)
            D[index, index - 1] = (l_tot[index] / c.dx / c.dx)

    # remove the additional columns and rows who belong to the additional left and right elements.
    D = np.delete(D, list, axis=1)
    D = np.delete(D, list, axis=0)

    l = np.zeros(((N - 2) * (M)))

    # only the first N-2 elements have inflow and the last N-2 due to outflow
    for i in range((N - 2) * M):
        if math.floor(i / (N - 2)) == 0:
            index = i % (N - 2) + 1
            l[i] = (0*dl_totdy[index] - 2 / c.dx * l_tot[index]) * c.u_inj / l_w(Sw[index], c)
        if math.floor(i / (N - 2)) == M - 1:
            index = (M - 1) * N + i % (N - 2) + 1
            l[i] = (0*dl_totdy[index] + 2 / c.dx * l_tot[index]) * c.u_inj / l_t(Sw[index], c)

    Swt = -1 / c.phi * np.matmul(D, p) - l
    Swt = Swt.reshape(M, N - 2)
    Swt[0, :] = np.zeros(N - 2)
    Swt[-1, :] = np.zeros(N - 2)
    return Swt

def calc_vel(Sw,c):
    M,N = Sw.shape
    N = N + 2

    # add periodic BC to Sw
    Sw = np.hstack([np.expand_dims(Sw[:,-1], axis=1),Sw,np.expand_dims(Sw[:,0], axis=1)])
    Sw = Sw.reshape(M*N)

    # prealocate memory
    l_tot  = np.zeros(N*M)
    l_wL   = np.zeros(N*M)
    l_oL   = np.zeros(N*M)
    dl_tot = np.zeros(N*M)
    dl_totdx = np.zeros(N*M)
    dl_totdy = np.zeros(N*M)

    # for all elements, calculate the following
    K = N*M
    for i in range(K):
        # calculate lambda_tot
        l_tot[i]   = l_t(Sw[i],c)
        l_wL[i]    = l_w(Sw[i],c)
        l_oL[i]    = l_o(Sw[i],c)
        dl_tot[i]  = dl_w(Sw[i],c)+dl_o(Sw[i],c)
    for i in range(K):
        # calculate position derivatives of l_tot,
        # since some BC are periodic they need to be calculated differently
        if i%N == N-1:
            dl_totdx[i] = 1 / 2 / c.dx * (l_tot[(i + 1-N)] - l_tot[(i - 1)])
        elif i%N == 0:
            dl_totdx[i] = 1 / 2 / c.dx * (l_tot[(i + 1)] - l_tot[(i - 1 + N)])
        else:
            dl_totdx[i] = 1 / 2 / c.dx * (l_tot[(i + 1)] - l_tot[(i - 1)])

        if math.floor(i/N) == M-1:
            dl_totdy[i] = 1 / c.dx * (3*l_tot[(i)] - 4 * l_tot[(i - N)] + l_tot[(i - 2 *N)])
        elif math.floor(i/N) == 0:
            # note that we have a less accurate calculation here
            dl_totdy[i] = 1 /2 / c.dx * (-l_tot[i+2*N]+ 4*l_tot[i+N] - 3* l_tot[(i)])
        else:
            dl_totdy[i] = 1 / 2 / c.dx * (l_tot[i + N] - l_tot[i-N])

        # set up diagonals
        diagonals = [-4 / c.dx / c.dx * l_tot,
                     l_tot[0:-1] / c.dx / c.dx + dl_totdx[0:-1] / 2 / c.dx,
                     l_tot[1:] / c.dx / c.dx - dl_totdx[1:] / 2 / c.dx,
                     l_tot[0:-N] / c.dx / c.dx + dl_totdy[0:-N] / 2 / c.dx,
                     l_tot[N:] / c.dx / c.dx - dl_totdy[N:] / 2 / c.dx]
        D = diags(diagonals, [0, 1, -1, N, -N]).toarray()

        # create a list to remove the additional left and right elements (effectively we reduce N by 2)
        list = []
        for i in range(K):
            if i % N == 0:
                list.append(i)
            elif i % N == N - 1:
                list.append(i)

        # due to the periodic BC, we have to add the correct items in to the matrix (these are now in the
        # extra left and right elements and after we remove these, we have to add them back in.
        for j in range(M):
            index = j * (N) + (N - 2)
            D[index, j * N + 1] = (l_tot[index] / c.dx / c.dx + dl_totdx[index] / 2 / c.dx)
            index = j * (N) + 1
            D[index, j * N + N - 2] = (l_tot[index] / c.dx / c.dx - dl_totdx[index] / 2 / c.dx)

        # Since we have Neumann BC, we can eliminate the some of the values and need to add others:
        # we can get rid of the p_y terms, but we have to use a different stencil for the laplacian.
        for i in range(N - 1):
            index = i
            D[index, index + N] = 2 * (l_tot[index] / c.dx / c.dx)
            D[index, index + 1] = (l_tot[index] / c.dx / c.dx)
            D[index, index - 1] = (l_tot[index] / c.dx / c.dx)
            index = i + (M - 1) * (N)
            D[index, index - N] = 2 * (l_tot[index] / c.dx / c.dx)
            D[index, index + 1] = (l_tot[index] / c.dx / c.dx)
            D[index, index - 1] = (l_tot[index] / c.dx / c.dx)


    # remove the additional columns and rows who belong to the additional left and right elements.
    D = np.delete(D,list,axis=1)
    D = np.delete(D,list,axis=0)

    plt.matshow(D)
    plt.show()

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
            l[i] = (dl_totdy[index] - 2/c.dx*l_tot[index])*c.u_inj/l_wL[index]
        if math.floor(i/(N-2)) == M-1:
            index = (M-1)*N + i % (N - 2) + 1
            l[i] = (dl_totdy[index] + 2/c.dx*l_tot[index])*c.u_inj/l_tot[index]

    # solve matrix vector product
    A = np.matmul(np.transpose(D),D)
    b = np.matmul(np.transpose(D),l)
    p = np.linalg.solve(A,b)

    residual = np.matmul(D,p)-l
    # plt.show()
    # print(np.sum(residual))
    return p