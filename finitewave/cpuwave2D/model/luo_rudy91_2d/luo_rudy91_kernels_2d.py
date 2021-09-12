from math import log, sqrt, exp
from numba import njit, prange

from finitewave.core.exception.exceptions import IncorrectWeightsShapeError
from finitewave.cpuwave2D.model.diffuse_kernels_2d \
    import diffuse_kernel_2d_iso, diffuse_kernel_2d_aniso, _parallel


@njit(parallel=_parallel)
def ionic_kernel_2d(u_new, u, m, h, j_, d, f, x, Cai_c, mesh, dt):
    Ko_c = 5.4
    Ki_c = 145
    Nai_c = 18
    Nao_c = 140
    Cao_c = 1.8

    R = 8.314
    T = 310  # 37 cels
    F = 96.5

    PR_NaK = 0.01833
    E_Na = (R*T/F)*log(Nao_c/Nai_c)

    n_i = u.shape[0]
    n_j = u.shape[1]

    for ii in prange(n_i*n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if mesh[i, j] != 1:
            continue

        I_Na = 23*pow(m[i, j], 3)*h[i, j]*j_[i, j]*(u[i, j]-E_Na)

        alpha_h = 0
        beta_h = 0
        beta_J = 0
        alpha_J = 0
        if u[i, j] >= -40.:
            alpha_h = 0
            beta_h = 1./(0.13*(1 + exp((u[i, j] + 10.66)/-11.1)))
            beta_J = 0.3*exp(-2.535*1e-07 *
                             u[i, j])/(1 + exp(-0.1*(u[i, j]+32)))
            alpha_J = 0.
        else:
            alpha_h = 0.135*exp((80+u[i, j])/-6.8)
            beta_h = 3.56*exp(0.079*u[i, j]) + \
                3.1*100000*exp(0.35*u[i, j])
            beta_J = 0.1212 * \
                exp(-0.01052*u[i, j])/(1+exp(-0.1378*(u[i, j]+40.14)))
            alpha_J = (-1.2714*100000*exp(0.2444*u[i, j])-3.474*1e-05*exp(-0.04391*u[i, j]))*(
                u[i, j]+37.78)/(1+exp(0.311*(u[i, j]+79.23)))

        alpha_m = 0.32*(u[i, j]+47.13)/(1-exp(-0.1*(u[i, j]+47.13)))
        beta_m = 0.08*exp(-u[i, j]/11)

        tau_m = 1./(alpha_m+beta_m)
        inf_m = alpha_m/(alpha_m + beta_m)
        m[i, j] += dt*(inf_m - m[i, j])/tau_m

        tau_h = 1./(alpha_h+beta_h)
        inf_h = alpha_h/(alpha_h + beta_h)
        h[i, j] += dt*(inf_h - h[i, j])/tau_h

        tau_J = 1./(alpha_J+beta_J)
        inf_J = alpha_J/(alpha_J + beta_J)
        j_[i, j] += dt*(inf_J - j_[i, j])/tau_J

        # Slow inward current:
        E_Si = 7.7-13.0287*log(Cai_c[i, j])
        I_Si = 0.045*d[i, j]*f[i, j]*(u[i, j]-E_Si)
        alpha_d = 0.095*exp(-0.01*(u[i, j]-5)) / \
            (1+exp(-0.072*(u[i, j]-5)))
        beta_d = 0.07*exp(-0.017*(u[i, j]+44)) / \
            (1+exp(0.05*(u[i, j]+44)))
        alpha_f = 0.012 * \
            exp(-0.008*(u[i, j]+28))/(1+exp(0.15*(u[i, j]+28)))
        beta_f = 0.0065*exp(-0.02*(u[i, j]+30)) / \
            (1+exp(-0.2*(u[i, j]+30)))
        Cai_c[i, j] += dt*(-0.0001*I_Si+0.07*(0.0001-Cai_c[i, j]))

        tau_d = 1./(alpha_d+beta_d)
        inf_d = alpha_d/(alpha_d + beta_d)
        d[i, j] += dt*(inf_d - d[i, j])/tau_d

        tau_f = 1./(alpha_f+beta_f)
        inf_f = alpha_f/(alpha_f + beta_f)
        f[i, j] += dt*(inf_f - f[i, j])/tau_f

        # Time-dependent potassium current
        E_K = (R*T/F)*log((Ko_c + PR_NaK*Nao_c)/(Ki_c + PR_NaK*Nai_c))

        G_K = 0.705*sqrt(Ko_c/5.4)

        Xi = 0
        if u[i, j] > -100:
            Xi = 2.837*(exp(0.04*(u[i, j]+77))-1) / \
                ((u[i, j]+77)*exp(0.04*(u[i, j]+35)))
        else:
            Xi = 1

        I_K = G_K*x[i, j]*Xi*(u[i, j]-E_K)

        alpha_x = 0.0005 * \
            exp(0.083*(u[i, j]+50))/(1+exp(0.057*(u[i, j]+50)))
        beta_x = 0.0013*exp(-0.06*(u[i, j]+20)) / \
            (1+exp(-0.04*(u[i, j]+20)))

        tau_x = 1./(alpha_x+beta_x)
        inf_x = alpha_x/(alpha_x + beta_x)
        x[i, j] += dt*(inf_x - x[i, j])/tau_x

        # Time-independent potassium current:
        E_K1 = (R*T/F)*log(Ko_c/Ki_c)

        alpha_K1 = 1.02/(1+exp(0.2385*(u[i, j]-E_K1-59.215)))
        beta_K1 = (0.49124*exp(0.08032*(u[i, j]-E_K1+5.476))+exp(
            0.06175*(u[i, j]-E_K1-594.31)))/(1+exp(-0.5143*(u[i, j]-E_K1+4.753)))

        K_1x = alpha_K1/(alpha_K1+beta_K1)

        G_K1 = 0.6047*sqrt(Ko_c/5.4)
        I_K1 = G_K1*K_1x*(u[i, j]-E_K1)

        # Plateau potassium current:
        E_Kp = E_K1
        K_p = 1./(1+exp((7.488-u[i, j])/5.98))
        I_Kp = 0.0183*K_p*(u[i, j]-E_Kp)

        # Background current:
        I_b = 0.03921*(u[i, j]+59.87)

        # Total time-independent potassium current:
        I_K1_T = I_K1 + I_Kp + I_b

        u_new[i, j] -= dt * (I_Na + I_Si + I_K1_T + I_K)


class LuoRudy91Kernels2D:
    def __init__(self):
        pass

    @staticmethod
    def get_diffuse_kernel(shape):
        if shape[-1] == 5:
            return diffuse_kernel_2d_iso
        if shape[-1] == 9:
            return diffuse_kernel_2d_aniso
        else:
            raise IncorrectWeightsShapeError(shape, 5, 9)

    @staticmethod
    def get_ionic_kernel():
        return ionic_kernel_2d
