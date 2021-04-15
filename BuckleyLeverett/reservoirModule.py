# Functions
def D_cap(S_w):
    return 0


def f_w(S_w):
    S_wn = (S_w - S_wc)/(1 - S_or - S_wc)
    l_w = kappa / mu_w * k_rw0 * (S_wn) ** n_w
    l_o = kappa / mu_o * k_ro0 * (1-S_wn) ** n_o
    return l_w / (l_w + l_o)


def df_dSw(S_w):
    C_o = k_ro0 / mu_o
    C_w = k_rw0 / mu_w
    S_wn = (S_w - S_wc) / (1 - S_or - S_wc)
    df_dSwn = - (C_w*( -n_w* C_o* S_wn**(-n_w-1) * (1-S_wn)**n_o - n_o*C_o*S_wn**(-n_w) * (1-S_wn)**(n_o-1)))/(C_o*S_wn**(-n_w) * (1-S_wn)**n_o + C_w)**2
    return df_dSwn / (1 - S_or - S_wc)


def bisection(f, startpoints, n):
    x1 = startpoints[0]
    x2 = startpoints[1]
    for i in range(n):
        x3 = (x1 + x2)/ 2
        if f(x3) > 0:
            x1 = x3
        else:
            x2 = x3
    return (x1 + x2) / 2


def magic_function(x):
    return df_dSw(x) - (f_w(x) - f_w(S_wc))/(x - S_wc)