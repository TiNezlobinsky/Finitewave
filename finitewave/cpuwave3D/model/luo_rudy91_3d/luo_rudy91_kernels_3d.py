from math import log, sqrt, exp, pow
from numba import njit, prange

from finitewave.core.exception.exceptions import IncorrectWeightsShapeError
from finitewave.cpuwave3D.model.diffuse_kernels_3d \
    import diffuse_kernel_3d_iso, diffuse_kernel_3d_aniso, _parallel


@njit(parallel=_parallel)
def ionic_kernel_3d(u_new, u, m, h, j_, d, f, x, Cai_c, mesh, dt):
    """
    Computes the ionic currents and updates the state variables in the 3D Luo-Rudy 1991 cardiac model.

    This function updates the membrane potential `u` and the gating variables `m`, `h`, `j_`, `d`, `f`, `x` based on
    the Luo-Rudy 1991 equations. It also updates the calcium concentration `Cai_c`.

    Parameters
    ----------
    u_new : np.ndarray
        Array to store the updated membrane potential.
    u : np.ndarray
        Array of the current membrane potential values.
    m : np.ndarray
        Array for the gating variable `m`.
    h : np.ndarray
        Array for the gating variable `h`.
    j_ : np.ndarray
        Array for the gating variable `j_`.
    d : np.ndarray
        Array for the gating variable `d`.
    f : np.ndarray
        Array for the gating variable `f`.
    x : np.ndarray
        Array for the gating variable `x`.
    Cai_c : np.ndarray
        Array for the intracellular calcium concentration.
    mesh : np.ndarray
        Mesh array indicating the tissue types.
    dt : float
        Time step for the simulation.

    Notes
    -----
    The function uses various constants and equations specific to the Luo-Rudy 1991 model to compute ionic currents and
    update the state variables. The results are stored in `u_new`, which represents the membrane potential at the next
    time step.
    """
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
    n_k = u.shape[2]

    for ii in prange(n_i*n_j*n_k):
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k
        if mesh[i, j, k] != 1:
            continue

        I_Na = 23*pow(m[i, j, k], 3)*h[i, j, k] * \
            j_[i, j, k]*(u[i, j, k]-E_Na)

        alpha_h = 0
        beta_h = 0
        beta_J = 0
        alpha_J = 0
        if u[i, j, k] >= -40.:
            alpha_h = 0
            beta_h = 1./(0.13*(1 + exp((u[i, j, k] + 10.66)/-11.1)))
            beta_J = 0.3 * \
                exp(-2.535*1e-07*u[i, j, k]) / \
                (1 + exp(-0.1*(u[i, j, k]+32)))
            alpha_J = 0.
        else:
            alpha_h = 0.135*exp((80+u[i, j, k])/-6.8)
            beta_h = 3.56*exp(0.079*u[i, j, k]) + \
                3.1*100000*exp(0.35*u[i, j, k])
            beta_J = 0.1212 * \
                exp(-0.01052*u[i, j, k]) / \
                (1+exp(-0.1378*(u[i, j, k]+40.14)))
            alpha_J = (-1.2714*100000*exp(0.2444*u[i, j, k])-3.474*1e-05*exp(-0.04391*u[i, j, k]))*(
                u[i, j, k]+37.78)/(1+exp(0.311*(u[i, j, k]+79.23)))

        alpha_m = 0.32*(u[i, j, k]+47.13)/(1-exp(-0.1*(u[i, j, k]+47.13)))
        beta_m = 0.08*exp(-u[i, j, k]/11)

        tau_m = 1./(alpha_m+beta_m)
        inf_m = alpha_m/(alpha_m + beta_m)
        m[i, j, k] += dt*(inf_m - m[i, j, k])/tau_m

        tau_h = 1./(alpha_h+beta_h)
        inf_h = alpha_h/(alpha_h + beta_h)
        h[i, j, k] += dt*(inf_h - h[i, j, k])/tau_h

        tau_J = 1./(alpha_J+beta_J)
        inf_J = alpha_J/(alpha_J + beta_J)
        j_[i, j, k] += dt*(inf_J - j_[i, j, k])/tau_J

        # Slow inward current:
        E_Si = 7.7-13.0287*log(Cai_c[i, j, k])
        I_Si = 0.045*d[i, j, k]*f[i, j, k]*(u[i, j, k]-E_Si)
        alpha_d = 0.095 * \
            exp(-0.01*(u[i, j, k]-5))/(1+exp(-0.072*(u[i, j, k]-5)))
        beta_d = 0.07*exp(-0.017*(u[i, j, k]+44)) / \
            (1+exp(0.05*(u[i, j, k]+44)))
        alpha_f = 0.012 * \
            exp(-0.008*(u[i, j, k]+28))/(1+exp(0.15*(u[i, j, k]+28)))
        beta_f = 0.0065 * \
            exp(-0.02*(u[i, j, k]+30))/(1+exp(-0.2*(u[i, j, k]+30)))
        Cai_c[i, j, k] += dt*(-0.0001*I_Si+0.07*(0.0001-Cai_c[i, j, k]))

        tau_d = 1./(alpha_d+beta_d)
        inf_d = alpha_d/(alpha_d + beta_d)
        d[i, j, k] += dt*(inf_d - d[i, j, k])/tau_d

        tau_f = 1./(alpha_f+beta_f)
        inf_f = alpha_f/(alpha_f + beta_f)
        f[i, j, k] += dt*(inf_f - f[i, j, k])/tau_f

        # Time-dependent potassium current
        E_K = (R*T/F)*log((Ko_c + PR_NaK*Nao_c)/(Ki_c + PR_NaK*Nai_c))

        G_K = 0.705*sqrt(Ko_c/5.4)

        Xi = 0
        if u[i, j, k] > -100:
            Xi = 2.837*(exp(0.04*(u[i, j, k]+77))-1) / \
                ((u[i, j, k]+77)*exp(0.04*(u[i, j, k]+35)))
        else:
            Xi = 1

        I_K = G_K*x[i, j, k]*Xi*(u[i, j, k]-E_K)

        alpha_x = 0.0005 * \
            exp(0.083*(u[i, j, k]+50))/(1+exp(0.057*(u[i, j, k]+50)))
        beta_x = 0.0013 * \
            exp(-0.06*(u[i, j, k]+20))/(1+exp(-0.04*(u[i, j, k]+20)))

        tau_x = 1./(alpha_x+beta_x)
        inf_x = alpha_x/(alpha_x + beta_x)
        x[i, j, k] += dt*(inf_x - x[i, j, k])/tau_x

        # Time-independent potassium current:
        E_K1 = (R*T/F)*log(Ko_c/Ki_c)

        alpha_K1 = 1.02/(1+exp(0.2385*(u[i, j, k]-E_K1-59.215)))
        beta_K1 = (0.49124*exp(0.08032*(u[i, j, k]-E_K1+5.476))+exp(
            0.06175*(u[i, j, k]-E_K1-594.31)))/(1+exp(-0.5143*(u[i, j, k]-E_K1+4.753)))

        K_1x = alpha_K1/(alpha_K1+beta_K1)

        G_K1 = 0.6047*sqrt(Ko_c/5.4)
        I_K1 = G_K1*K_1x*(u[i, j, k]-E_K1)

        # Plateau potassium current:
        E_Kp = E_K1
        K_p = 1./(1+exp((7.488-u[i, j, k])/5.98))
        I_Kp = 0.0183*K_p*(u[i, j, k]-E_Kp)

        # Background current:
        I_b = 0.03921*(u[i, j, k]+59.87)

        # Total time-independent potassium current:
        I_K1_T = I_K1 + I_Kp + I_b

        u_new[i, j, k] += dt * (I_Na + I_Si + I_K1_T + I_K)


class LuoRudy91Kernels3D:
    """
    Class to handle kernel functions for the Luo-Rudy 1991 cardiac model in 3D.

    This class provides methods to obtain the appropriate diffusion and ionic kernels based on the shape of the weight array.

    Methods
    -------
    get_diffuse_kernel(shape):
        Returns the diffusion kernel function based on the weight array shape.
    get_ionic_kernel():
        Returns the ionic kernel function used for updating membrane potentials and gating variables.
    """
    def __init__(self):
        """
        Initializes the LuoRudy91Kernels3D instance.
        """
        pass

    @staticmethod
    def get_diffuse_kernel(shape):
        """
        Retrieves the diffusion kernel function based on the weight shape.

        Parameters
        ----------
        shape : tuple
            The shape of the weight array used in the diffusion process.

        Returns
        -------
        function
            The diffusion kernel function appropriate for the given weight shape.

        Raises
        ------
        IncorrectWeightsShapeError
            If the shape of the weights array does not match expected values (7 or 19).
        """
        if shape[-1] == 7:
            return diffuse_kernel_3d_iso
        if shape[-1] == 19:
            return diffuse_kernel_3d_aniso
        else:
            raise IncorrectWeightsShapeError(shape, 7, 19)

    @staticmethod
    def get_ionic_kernel():
        """
        Retrieves the ionic kernel function for updating membrane potentials and gating variables.

        Returns
        -------
        function
            The ionic kernel function used in the Luo-Rudy 1991 model.
        """
        return ionic_kernel_3d
