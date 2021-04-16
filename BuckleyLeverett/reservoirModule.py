from math import sqrt

class Constants:
    def __init__(self, phi, u_inj, mu_w, mu_o, kappa, k_rw0, k_ro0, n_w, n_o, S_or, S_wc, sigma,labda,dx):
        self.phi = phi
        self.u_inj = u_inj
        self.mu_w = mu_w
        self.mu_o = mu_o
        self.kappa = kappa
        self.k_rw0 = k_rw0
        self.k_ro0 = k_ro0
        self.n_w = n_w
        self.n_o = n_o
        self.S_or = S_or  # Oil rest-saturation
        self.S_wc = S_wc  # Water capillary saturation
        self.sigma = sigma
        self.labda = labda
        self.dx = dx

# Functions
def D_cap(S_w,c):
    return c.sigma*sqrt(c.phi/c.kappa)*S_w**(-1/c.labda)


def f_w(S_w, c):
    S_wn = (S_w - c.S_wc)/(1 - c.S_or - c.S_wc)
    l_w = c.kappa / c.mu_w * c.k_rw0 * (S_wn) ** c.n_w
    l_o = c.kappa / c.mu_o * c.k_ro0 * (1-S_wn) ** c.n_o
    return l_w / (l_w + l_o)

def l_w(S_w, c):
    S_wn = (S_w - c.S_wc) / (1 - c.S_or - c.S_wc)
    return c.kappa / c.mu_w * c.k_rw0 * (S_wn) ** c.n_w

def dl_w(S_w, c):
    S_wn = (S_w - c.S_wc) / (1 - c.S_or - c.S_wc)
    return c.kappa / c.mu_w * c.k_rw0 * c.n_w * (S_wn) ** (c.n_w-1) / (1 - c.S_or - c.S_wc)

def l_o(S_w, c):
    S_wn = (S_w - c.S_wc) / (1 - c.S_or - c.S_wc)
    return c.kappa / c.mu_o * c.k_ro0 *  (1-S_wn) ** c.n_o

def dl_o(S_w, c):
    S_wn = (S_w - c.S_wc) / (1 - c.S_or - c.S_wc)
    return c.kappa / c.mu_o * c.k_ro0 * c.n_o * (1-S_wn) ** (c.n_o - 1) / (1 - c.S_or - c.S_wc)

def l_t(S_w,c):
    return l_w(S_w,c) + l_o(S_w,c)

def df_dSw(S_w, c):
    C_o = c.k_ro0 / c.mu_o
    C_w = c.k_rw0 / c.mu_w
    S_wn = (S_w - c.S_wc) / (1 - c.S_or - c.S_wc)
    df_dSwn = - (C_w*( -c.n_w* C_o* S_wn**(-c.n_w-1) * (1-S_wn)**c.n_o - c.n_o*C_o*S_wn**(-c.n_w) * (1-S_wn)**(c.n_o-1)))/(C_o*S_wn**(-c.n_w) * (1-S_wn)**c.n_o + C_w)**2
    return df_dSwn / (1 - c.S_or - c.S_wc)


def bisection(f, startpoints, n, c):
    x1 = startpoints[0]
    x2 = startpoints[1]
    for i in range(n):
        x3 = (x1 + x2)/ 2
        if f(x3, c) > 0:
            x1 = x3
        else:
            x2 = x3
    return (x1 + x2) / 2


# def magic_function(x, c):
#     return df_dSw(x) - (f_w(x) - f_w(S_wc))/(x - c.S_wc)

