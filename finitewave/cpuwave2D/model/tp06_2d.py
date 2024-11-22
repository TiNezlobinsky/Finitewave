import numpy as np
from numba import njit, prange

from finitewave.core.model.cardiac_model import CardiacModel
from finitewave.cpuwave2D.stencil.asymmetric_stencil_2d import (
    AsymmetricStencil2D
)
from finitewave.cpuwave2D.stencil.isotropic_stencil_2d import (
    IsotropicStencil2D
)


class TP062D(CardiacModel):
    """
    A class to represent the TP06 cardiac model in 2D.

    Attributes
    ----------
    state_vars : list of str
        List of state variable names.
    """

    def __init__(self):
        """
        Initializes the TP062D cardiac model.

        Sets up the arrays for state variables and model parameters.
        """
        super().__init__()
        self.D_model = 0.154
        self.state_vars = ["u", "Cai", "CaSR", "CaSS", "Nai", "Ki",
                           "M_", "H_", "J_", "Xr1", "Xr2", "Xs", "R_",
                           "S_", "D_", "F_", "F2_", "FCass", "RR", "OO"]

    def initialize(self):
        """
        Initializes the model's state variables and diffusion/ionic kernels.

        Sets up the initial values for membrane potential, ion concentrations,
        gating variables, and assigns the appropriate kernel functions.
        """
        super().initialize()
        shape = self.cardiac_tissue.mesh.shape

        self.u = -84.5*np.ones(shape, dtype=self.npfloat)
        self.u_new = self.u.copy()
        self.Cai = 0.00007*np.ones(shape, dtype=self.npfloat)
        self.CaSR = 1.3*np.ones(shape, dtype=self.npfloat)
        self.CaSS = 0.00007*np.ones(shape, dtype=self.npfloat)
        self.Nai = 7.67*np.ones(shape, dtype=self.npfloat)
        self.Ki = 138.3*np.ones(shape, dtype=self.npfloat)
        self.M_ = np.zeros(shape, dtype=self.npfloat)
        self.H_ = 0.75*np.ones(shape, dtype=self.npfloat)
        self.J_ = 0.75*np.ones(shape, dtype=self.npfloat)
        self.Xr1 = np.zeros(shape, dtype=self.npfloat)
        self.Xr2 = np.ones(shape, dtype=self.npfloat)
        self.Xs = np.zeros(shape, dtype=self.npfloat)
        self.R_ = np.zeros(shape, dtype=self.npfloat)
        self.S_ = np.ones(shape, dtype=self.npfloat)
        self.D_ = np.zeros(shape, dtype=self.npfloat)
        self.F_ = np.ones(shape, dtype=self.npfloat)
        self.F2_ = np.ones(shape, dtype=self.npfloat)
        self.FCass = np.ones(shape, dtype=self.npfloat)
        self.RR = np.ones(shape, dtype=self.npfloat)
        self.OO = np.zeros(shape, dtype=self.npfloat)

    def run_ionic_kernel(self):
        """
        Executes the ionic kernel function to update ionic currents and state
        variables
        """
        ionic_kernel_2d(self.u_new, self.u, self.Cai, self.CaSR, self.CaSS,
                        self.Nai, self.Ki, self.M_, self.H_, self.J_, self.Xr1,
                        self.Xr2, self.Xs, self.R_, self.S_, self.D_, self.F_,
                        self.F2_, self.FCass, self.RR, self.OO,
                        self.cardiac_tissue.mesh, self.dt)

    def select_stencil(self, cardiac_tissue):
        """
        Selects the appropriate stencil for diffusion based on the tissue
        properties. If the tissue has fiber directions, an asymmetric stencil
        is used; otherwise, an isotropic stencil is used.

        Parameters
        ----------
        cardiac_tissue : CardiacTissue2D
            A tissue object representing the cardiac tissue.

        Returns
        -------
        Stencil
            The stencil object to use for diffusion computations.
        """
        if cardiac_tissue.fibers is None:
            return IsotropicStencil2D()

        return AsymmetricStencil2D()


# tp06 epi kernel
@njit(parallel=True)
def ionic_kernel_2d(u_new, u, Cai, CaSR, CaSS, Nai, Ki, M_, H_, J_, Xr1, Xr2,
                    Xs, R_, S_, D_, F_, F2_, FCass, RR, OO, mesh, dt):
    """
    Compute the ionic currents and update the state variables for the 2D TP06
    cardiac model.

    This function calculates the ionic currents based on the TP06 cardiac
    model, updates ion concentrations, and modifies gating variables in the
    2D grid. The calculations are performed in parallel to enhance performance.

    Parameters
    ----------
    u_new : numpy.ndarray
        Array to store the updated membrane potential values.
    u : numpy.ndarray
        Array of current membrane potential values.
    Cai : numpy.ndarray
        Array of calcium concentration in the cytosol.
    CaSR : numpy.ndarray
        Array of calcium concentration in the sarcoplasmic reticulum.
    CaSS : numpy.ndarray
        Array of calcium concentration in the submembrane space.
    Nai : numpy.ndarray
        Array of sodium ion concentration in the intracellular space.
    Ki : numpy.ndarray
        Array of potassium ion concentration in the intracellular space.
    M_ : numpy.ndarray
        Array of gating variable for sodium channels (activation).
    H_ : numpy.ndarray
        Array of gating variable for sodium channels (inactivation).
    J_ : numpy.ndarray
        Array of gating variable for sodium channels (inactivation).
    Xr1 : numpy.ndarray
        Array of gating variable for rapid delayed rectifier potassium
        channels.
    Xr2 : numpy.ndarray
        Array of gating variable for rapid delayed rectifier potassium
        channels.
    Xs : numpy.ndarray
        Array of gating variable for slow delayed rectifier potassium channels.
    R_ : numpy.ndarray
        Array of gating variable for ryanodine receptors.
    S_ : numpy.ndarray
        Array of gating variable for calcium-sensitive current.
    D_ : numpy.ndarray
        Array of gating variable for L-type calcium channels.
    F_ : numpy.ndarray
        Array of gating variable for calcium-dependent calcium channels.
    F2_ : numpy.ndarray
        Array of secondary gating variable for calcium-dependent calcium
        channels.
    FCass : numpy.ndarray
        Array of gating variable for calcium-sensitive current.
    RR : numpy.ndarray
        Array of ryanodine receptor gating variable for calcium release.
    OO : numpy.ndarray
        Array of ryanodine receptor gating variable for calcium release.
    mesh : numpy.ndarray
        Mesh grid indicating tissue areas.
    dt : float
        Time step for the simulation.

    Returns
    -------
    None
        The function updates the state variables in place. No return value is
        produced.
    """
    n_i = u.shape[0]
    n_j = u.shape[1]

    # Needed to compute currents
    Ko = 5.4
    Cao = 2.0
    Nao = 140.0

    Vc = 0.016404
    Vsr = 0.001094
    Vss = 0.00005468

    Bufc = 0.2
    Kbufc = 0.001
    Bufsr = 10.
    Kbufsr = 0.3
    Bufss = 0.4
    Kbufss = 0.00025

    Vmaxup = 0.006375
    Kup = 0.00025
    Vrel = 0.102  # 40.8
    k1_ = 0.15
    k2_ = 0.045
    k3 = 0.060
    k4 = 0.005  # 0.000015
    EC = 1.5
    maxsr = 2.5
    minsr = 1.
    Vleak = 0.00036
    Vxfer = 0.0038

    R = 8314.472
    F = 96485.3415
    T = 310.0
    RTONF = 26.713760659695648

    CAPACITANCE = 0.185

    Gkr = 0.153

    pKNa = 0.03

    GK1 = 5.405

    GNa = 14.838

    GbNa = 0.00029

    KmK = 1.0
    KmNa = 40.0
    knak = 2.724

    GCaL = 0.00003980

    GbCa = 0.000592

    knaca = 1000
    KmNai = 87.5
    KmCa = 1.38
    ksat = 0.1
    n_ = 0.35

    GpCa = 0.1238
    KpCa = 0.0005

    GpK = 0.0146

    Gto = 0.294
    Gks = 0.392

    inverseVcF2 = 1./(2*Vc*F)
    inverseVcF = 1./(Vc*F)
    inversevssF2 = 1./(2*Vss*F)

    for ii in prange(n_i*n_j):
        i = int(ii/n_j)
        j = ii % n_j
        if mesh[i, j] != 1:
            continue

        Ek = RTONF*(np.log((Ko/Ki[i, j])))
        Ena = RTONF*(np.log((Nao/Nai[i, j])))
        Eks = RTONF*(np.log((Ko+pKNa*Nao)/(Ki[i, j]+pKNa*Nai[i, j])))
        Eca = 0.5*RTONF*(np.log((Cao/Cai[i, j])))
        Ak1 = 0.1/(1.+np.exp(0.06*(u[i, j]-Ek-200)))
        Bk1 = (3.*np.exp(0.0002*(u[i, j]-Ek+100)) +
               np.exp(0.1*(u[i, j]-Ek-10)))/(1.+np.exp(-0.5*(u[i, j]-Ek)))
        rec_iK1 = Ak1/(Ak1+Bk1)
        rec_iNaK = (
            1./(1.+0.1245*np.exp(-0.1*u[i, j]*F/(R*T))+0.0353*np.exp(-u[i, j]*F/(R*T))))
        rec_ipK = 1./(1.+np.exp((25-u[i, j])/5.98))

        # Compute currents
        INa = GNa*M_[i, j]*M_[i, j]*M_[i, j]*H_[i, j]*J_[i, j]*(u[i, j]-Ena)
        ICaL = GCaL*D_[i, j]*F_[i, j]*F2_[i, j]*FCass[i, j]*4*(u[i, j]-15)*(F*F/(R*T)) *\
            (0.25*np.exp(2*(u[i, j]-15)*F/(R*T))*CaSS[i, j]-Cao) / \
            (np.exp(2*(u[i, j]-15)*F/(R*T))-1.)
        Ito = Gto*R_[i, j]*S_[i, j]*(u[i, j]-Ek)
        IKr = Gkr*np.sqrt(Ko/5.4)*Xr1[i, j]*Xr2[i, j]*(u[i, j]-Ek)
        IKs = Gks*Xs[i, j]*Xs[i, j]*(u[i, j]-Eks)
        IK1 = GK1*rec_iK1*(u[i, j]-Ek)
        INaCa = knaca*(1./(KmNai*KmNai*KmNai+Nao*Nao*Nao))*(1./(KmCa+Cao)) *\
            (1./(1+ksat*np.exp((n_-1)*u[i, j]*F/(R*T)))) *\
            (np.exp(n_*u[i, j]*F/(R*T))*Nai[i, j]*Nai[i, j]*Nai[i, j]*Cao -
                np.exp((n_-1)*u[i, j]*F/(R*T))*Nao*Nao*Nao*Cai[i, j]*2.5)
        INaK = knak*(Ko/(Ko+KmK))*(Nai[i, j]/(Nai[i, j]+KmNa))*rec_iNaK
        IpCa = GpCa*Cai[i, j]/(KpCa+Cai[i, j])
        IpK = GpK*rec_ipK*(u[i, j]-Ek)
        IbNa = GbNa*(u[i, j]-Ena)
        IbCa = GbCa*(u[i, j]-Eca)

        # Determine total current
        u_new[i, j] -= dt * (IKr + IKs + IK1 + Ito + INa +
                             IbNa + ICaL + IbCa + INaK + INaCa + IpCa + IpK)

        # update concentrations
        kCaSR = maxsr-((maxsr-minsr)/(1+(EC/CaSR[i, j])*(EC/CaSR[i, j])))
        k1 = k1_/kCaSR
        k2 = k2_*kCaSR
        dRR = k4*(1-RR[i, j])-k2*CaSS[i, j]*RR[i, j]
        RR[i, j] += dt*dRR
        OO[i, j] = k1*CaSS[i, j]*CaSS[i, j] * \
            RR[i, j]/(k3+k1*CaSS[i, j]*CaSS[i, j])

        Irel = Vrel*OO[i, j]*(CaSR[i, j]-CaSS[i, j])
        Ileak = Vleak*(CaSR[i, j]-Cai[i, j])
        Iup = Vmaxup/(1.+((Kup*Kup)/(Cai[i, j]*Cai[i, j])))
        Ixfer = Vxfer*(CaSS[i, j]-Cai[i, j])

        CaCSQN = Bufsr*CaSR[i, j]/(CaSR[i, j]+Kbufsr)
        dCaSR = dt*(Iup-Irel-Ileak)
        bjsr = Bufsr-CaCSQN-dCaSR-CaSR[i, j]+Kbufsr
        cjsr = Kbufsr*(CaCSQN+dCaSR+CaSR[i, j])
        CaSR[i, j] = (np.sqrt(bjsr*bjsr+4*cjsr)-bjsr)/2

        CaSSBuf = Bufss*CaSS[i, j]/(CaSS[i, j]+Kbufss)
        dCaSS = dt*(-Ixfer*(Vc/Vss)+Irel*(Vsr/Vss) +
                    (-ICaL*inversevssF2*CAPACITANCE))
        bcss = Bufss-CaSSBuf-dCaSS-CaSS[i, j]+Kbufss
        ccss = Kbufss*(CaSSBuf+dCaSS+CaSS[i, j])
        CaSS[i, j] = (np.sqrt(bcss*bcss+4*ccss)-bcss)/2

        CaBuf = Bufc*Cai[i, j]/(Cai[i, j]+Kbufc)
        dCai = dt*((-(IbCa+IpCa-2*INaCa)*inverseVcF2*CAPACITANCE) -
                   (Iup-Ileak)*(Vsr/Vc)+Ixfer)
        bc = Bufc-CaBuf-dCai-Cai[i, j]+Kbufc
        cc = Kbufc*(CaBuf+dCai+Cai[i, j])
        Cai[i, j] = (np.sqrt(bc*bc+4*cc)-bc)/2

        dNai = -(INa+IbNa+3*INaK+3*INaCa)*inverseVcF*CAPACITANCE
        Nai[i, j] += dt*dNai

        dKi = -(IK1+Ito+IKr+IKs-2*INaK+IpK)*inverseVcF*CAPACITANCE
        Ki[i, j] += dt*dKi

        # compute steady state values and time constants
        AM = 1./(1.+np.exp((-60.-u[i, j])/5.))
        BM = 0.1/(1.+np.exp((u[i, j]+35.)/5.)) + \
            0.10/(1.+np.exp((u[i, j]-50.)/200.))
        TAU_M = AM*BM
        M_INF = 1./((1.+np.exp((-56.86-u[i, j])/9.03))
                    * (1.+np.exp((-56.86-u[i, j])/9.03)))

        AH_ = 0.
        BH_ = 0.
        if u[i, j] >= -40.:
            AH_ = 0.
            BH_ = 0.77/(0.13*(1.+np.exp(-(u[i, j]+10.66)/11.1)))
        else:
            AH_ = 0.057*np.exp(-(u[i, j]+80.)/6.8)
            BH_ = 2.7*np.exp(0.079*u[i, j])+(3.1e5)*np.exp(0.3485*u[i, j])

        TAU_H = 1.0/(AH_ + BH_)

        H_INF = 1./((1.+np.exp((u[i, j]+71.55)/7.43))
                    * (1.+np.exp((u[i, j]+71.55)/7.43)))

        AJ_ = 0.
        BJ_ = 0.
        if u[i, j] >= -40.:
            AJ_ = 0.
            BJ_ = 0.6*np.exp((0.057)*u[i, j])/(1.+np.exp(-0.1*(u[i, j]+32.)))
        else:
            AJ_ = ((-2.5428e4)*np.exp(0.2444*u[i, j])-(6.948e-6) *
                   np.exp(-0.04391*u[i, j]))*(u[i, j]+37.78) /\
                (1.+np.exp(0.311*(u[i, j]+79.23)))
            BJ_ = 0.02424*np.exp(-0.01052*u[i, j]) / \
                (1.+np.exp(-0.1378*(u[i, j]+40.14)))

        TAU_J = 1.0/(AJ_ + BJ_)

        J_INF = H_INF

        Xr1_INF = 1./(1.+np.exp((-26.-u[i, j])/7.))
        axr1 = 450./(1.+np.exp((-45.-u[i, j])/10.))
        bxr1 = 6./(1.+np.exp((u[i, j]-(-30.))/11.5))
        TAU_Xr1 = axr1*bxr1
        Xr2_INF = 1./(1.+np.exp((u[i, j]-(-88.))/24.))
        axr2 = 3./(1.+np.exp((-60.-u[i, j])/20.))
        bxr2 = 1.12/(1.+np.exp((u[i, j]-60.)/20.))
        TAU_Xr2 = axr2*bxr2

        Xs_INF = 1./(1.+np.exp((-5.-u[i, j])/14.))
        Axs = (1400./(np.sqrt(1.+np.exp((5.-u[i, j])/6))))
        Bxs = (1./(1.+np.exp((u[i, j]-35.)/15.)))
        TAU_Xs = Axs*Bxs+80

        R_INF = 0
        S_INF = 0
        TAU_R = 0
        TAU_S = 0

        R_INF = 1./(1.+np.exp((20-u[i, j])/6.))
        S_INF = 1./(1.+np.exp((u[i, j]+20)/5.))
        TAU_R = 9.5*np.exp(-(u[i, j]+40.)*(u[i, j]+40.)/1800.)+0.8
        TAU_S = 85.*np.exp(-(u[i, j]+45.)*(u[i, j]+45.)/320.) + \
            5./(1.+np.exp((u[i, j]-20.)/5.))+3.

        D_INF = 1./(1.+np.exp((-8-u[i, j])/7.5))
        Ad = 1.4/(1.+np.exp((-35-u[i, j])/13))+0.25
        Bd = 1.4/(1.+np.exp((u[i, j]+5)/5))
        Cd = 1./(1.+np.exp((50-u[i, j])/20))
        TAU_D = Ad*Bd+Cd
        F_INF = 1./(1.+np.exp((u[i, j]+20)/7))
        Af = 1102.5*np.exp(-(u[i, j]+27)*(u[i, j]+27)/225)
        Bf = 200./(1+np.exp((13-u[i, j])/10.))
        Cf = (180./(1+np.exp((u[i, j]+30)/10)))+20
        TAU_F = Af+Bf+Cf
        F2_INF = 0.67/(1.+np.exp((u[i, j]+35)/7))+0.33
        Af2 = 600*np.exp(-(u[i, j]+25)*(u[i, j]+25)/170)
        Bf2 = 31/(1.+np.exp((25-u[i, j])/10))
        Cf2 = 16/(1.+np.exp((u[i, j]+30)/10))
        TAU_F2 = Af2+Bf2+Cf2
        FCaSS_INF = 0.6/(1+(CaSS[i, j]/0.05)*(CaSS[i, j]/0.05))+0.4
        TAU_FCaSS = 80./(1+(CaSS[i, j]/0.05)*(CaSS[i, j]/0.05))+2.

        # Update gates
        M_[i, j] = M_INF-(M_INF-M_[i, j])*np.exp(-dt/TAU_M)
        H_[i, j] = H_INF-(H_INF-H_[i, j])*np.exp(-dt/TAU_H)
        J_[i, j] = J_INF-(J_INF-J_[i, j])*np.exp(-dt/TAU_J)
        Xr1[i, j] = Xr1_INF-(Xr1_INF-Xr1[i, j])*np.exp(-dt/TAU_Xr1)
        Xr2[i, j] = Xr2_INF-(Xr2_INF-Xr2[i, j])*np.exp(-dt/TAU_Xr2)
        Xs[i, j] = Xs_INF-(Xs_INF-Xs[i, j])*np.exp(-dt/TAU_Xs)
        S_[i, j] = S_INF-(S_INF-S_[i, j])*np.exp(-dt/TAU_S)
        R_[i, j] = R_INF-(R_INF-R_[i, j])*np.exp(-dt/TAU_R)
        D_[i, j] = D_INF-(D_INF-D_[i, j])*np.exp(-dt/TAU_D)
        F_[i, j] = F_INF-(F_INF-F_[i, j])*np.exp(-dt/TAU_F)
        F2_[i, j] = F2_INF-(F2_INF-F2_[i, j])*np.exp(-dt/TAU_F2)
        FCass[i, j] = FCaSS_INF-(FCaSS_INF-FCass[i, j])*np.exp(-dt/TAU_FCaSS)
