from math import log, sqrt, exp
from numba import njit, prange
from finitewave.cpuwave3D.model.diffuse_kernels_3d \
    import diffuse_kernel_3d_iso, diffuse_kernel_3d_aniso, _parallel


# tp06 epi kernel
@njit(parallel=_parallel)
def ionic_kernel_3d(u_new, u, Cai, CaSR, CaSS, Nai, Ki, M_, H_, J_, Xr1, Xr2,
                    Xs, R_, S_, D_, F_, F2_, FCass, RR, OO, mesh, dt):
    n_i = u.shape[0]
    n_j = u.shape[1]
    n_k = u.shape[2]
    for ii in prange(n_i*n_j*n_k):
        i = ii//(n_j*n_k)
        j = (ii % (n_j*n_k))//n_k
        k = (ii % (n_j*n_k)) % n_k
        if mesh[i, j, k] != 1:
            continue

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

        Ek = RTONF*(log((Ko/Ki[i, j, k])))
        Ena = RTONF*(log((Nao/Nai[i, j, k])))
        Eks = RTONF*(log((Ko+pKNa*Nao)/(Ki[i, j, k]+pKNa*Nai[i, j, k])))
        Eca = 0.5*RTONF*(log((Cao/Cai[i, j, k])))
        Ak1 = 0.1/(1.+exp(0.06*(u[i, j, k]-Ek-200)))
        Bk1 = (3.*exp(0.0002*(u[i, j, k]-Ek+100)) +
               exp(0.1*(u[i, j, k]-Ek-10)))/(1.+exp(-0.5*(u[i, j, k]-Ek)))
        rec_iK1 = Ak1/(Ak1+Bk1)
        rec_iNaK = (
            1./(1.+0.1245*exp(-0.1*u[i, j, k]*F/(R*T))+0.0353*exp(-u[i, j, k]*F/(R*T))))
        rec_ipK = 1./(1.+exp((25-u[i, j, k])/5.98))

        # Compute currents
        INa = GNa*M_[i, j, k]*M_[i, j, k]*M_[i, j, k] * \
            H_[i, j, k]*J_[i, j, k]*(u[i, j, k]-Ena)
        ICaL = GCaL*D_[i, j, k]*F_[i, j, k]*F2_[i, j, k]*FCass[i, j, k]*4*(u[i, j, k]-15)*(F*F/(R*T)) *\
            (0.25*exp(2*(u[i, j, k]-15)*F/(R*T))*CaSS[i, j, k] -
             Cao)/(exp(2*(u[i, j, k]-15)*F/(R*T))-1.)
        Ito = Gto*R_[i, j, k]*S_[i, j, k]*(u[i, j, k]-Ek)
        IKr = Gkr*sqrt(Ko/5.4)*Xr1[i, j, k]*Xr2[i, j, k]*(u[i, j, k]-Ek)
        IKs = Gks*Xs[i, j, k]*Xs[i, j, k]*(u[i, j, k]-Eks)
        IK1 = GK1*rec_iK1*(u[i, j, k]-Ek)
        INaCa = knaca*(1./(KmNai*KmNai*KmNai+Nao*Nao*Nao))*(1./(KmCa+Cao)) *\
            (1./(1+ksat*exp((n_-1)*u[i, j, k]*F/(R*T)))) *\
            (exp(n_*u[i, j, k]*F/(R*T))*Nai[i, j, k]*Nai[i, j, k]*Nai[i, j, k]*Cao -
                exp((n_-1)*u[i, j, k]*F/(R*T))*Nao*Nao*Nao*Cai[i, j, k]*2.5)
        INaK = knak*(Ko/(Ko+KmK))*(Nai[i, j, k]/(Nai[i, j, k]+KmNa))*rec_iNaK
        IpCa = GpCa*Cai[i, j, k]/(KpCa+Cai[i, j, k])
        IpK = GpK*rec_ipK*(u[i, j, k]-Ek)
        IbNa = GbNa*(u[i, j, k]-Ena)
        IbCa = GbCa*(u[i, j, k]-Eca)

        # Determine total current
        u_new[i, j, k] -= dt * (IKr + IKs + IK1 + Ito + INa +
                                IbNa + ICaL + IbCa + INaK + INaCa + IpCa + IpK)

        # update concentrations
        kCaSR = maxsr-((maxsr-minsr)/(1+(EC/CaSR[i, j, k])*(EC/CaSR[i, j, k])))
        k1 = k1_/kCaSR
        k2 = k2_*kCaSR
        dRR = k4*(1-RR[i, j, k])-k2*CaSS[i, j, k]*RR[i, j, k]
        RR[i, j, k] += dt*dRR
        OO[i, j, k] = k1*CaSS[i, j, k]*CaSS[i, j, k] * \
            RR[i, j, k]/(k3+k1*CaSS[i, j, k]*CaSS[i, j, k])

        Irel = Vrel*OO[i, j, k]*(CaSR[i, j, k]-CaSS[i, j, k])
        Ileak = Vleak*(CaSR[i, j, k]-Cai[i, j, k])
        Iup = Vmaxup/(1.+((Kup*Kup)/(Cai[i, j, k]*Cai[i, j, k])))
        Ixfer = Vxfer*(CaSS[i, j, k]-Cai[i, j, k])

        CaCSQN = Bufsr*CaSR[i, j, k]/(CaSR[i, j, k]+Kbufsr)
        dCaSR = dt*(Iup-Irel-Ileak)
        bjsr = Bufsr-CaCSQN-dCaSR-CaSR[i, j, k]+Kbufsr
        cjsr = Kbufsr*(CaCSQN+dCaSR+CaSR[i, j, k])
        CaSR[i, j, k] = (sqrt(bjsr*bjsr+4*cjsr)-bjsr)/2

        CaSSBuf = Bufss*CaSS[i, j, k]/(CaSS[i, j, k]+Kbufss)
        dCaSS = dt*(-Ixfer*(Vc/Vss)+Irel*(Vsr/Vss) +
                    (-ICaL*inversevssF2*CAPACITANCE))
        bcss = Bufss-CaSSBuf-dCaSS-CaSS[i, j, k]+Kbufss
        ccss = Kbufss*(CaSSBuf+dCaSS+CaSS[i, j, k])
        CaSS[i, j, k] = (sqrt(bcss*bcss+4*ccss)-bcss)/2

        CaBuf = Bufc*Cai[i, j, k]/(Cai[i, j, k]+Kbufc)
        dCai = dt*((-(IbCa+IpCa-2*INaCa)*inverseVcF2*CAPACITANCE) -
                   (Iup-Ileak)*(Vsr/Vc)+Ixfer)
        bc = Bufc-CaBuf-dCai-Cai[i, j, k]+Kbufc
        cc = Kbufc*(CaBuf+dCai+Cai[i, j, k])
        Cai[i, j, k] = (sqrt(bc*bc+4*cc)-bc)/2

        dNai = -(INa+IbNa+3*INaK+3*INaCa)*inverseVcF*CAPACITANCE
        Nai[i, j, k] += dt*dNai

        dKi = -(IK1+Ito+IKr+IKs-2*INaK+IpK)*inverseVcF*CAPACITANCE
        Ki[i, j, k] += dt*dKi

        # compute steady state values and time constants
        AM = 1./(1.+exp((-60.-u[i, j, k])/5.))
        BM = 0.1/(1.+exp((u[i, j, k]+35.)/5.)) + \
            0.10/(1.+exp((u[i, j, k]-50.)/200.))
        TAU_M = AM*BM
        M_INF = 1./((1.+exp((-56.86-u[i, j, k])/9.03))
                    * (1.+exp((-56.86-u[i, j, k])/9.03)))

        AH_ = 0.
        BH_ = 0.
        if u[i, j, k] >= -40.:
            AH_ = 0.
            BH_ = 0.77/(0.13*(1.+exp(-(u[i, j, k]+10.66)/11.1)))
        else:
            AH_ = 0.057*exp(-(u[i, j, k]+80.)/6.8)
            BH_ = 2.7*exp(0.079*u[i, j, k])+(3.1e5)*exp(0.3485*u[i, j, k])

        TAU_H = 1.0/(AH_ + BH_)

        H_INF = 1./((1.+exp((u[i, j, k]+71.55)/7.43))
                    * (1.+exp((u[i, j, k]+71.55)/7.43)))

        AJ_ = 0.
        BJ_ = 0.
        if u[i, j, k] >= -40.:
            AJ_ = 0.
            BJ_ = 0.6*exp((0.057)*u[i, j, k])/(1.+exp(-0.1*(u[i, j, k]+32.)))
        else:
            AJ_ = ((-2.5428e4)*exp(0.2444*u[i, j, k])-(6.948e-6) *
                   exp(-0.04391*u[i, j, k]))*(u[i, j, k]+37.78) /\
                (1.+exp(0.311*(u[i, j, k]+79.23)))
            BJ_ = 0.02424*exp(-0.01052*u[i, j, k]) / \
                (1.+exp(-0.1378*(u[i, j, k]+40.14)))

        TAU_J = 1.0/(AJ_ + BJ_)

        J_INF = H_INF

        Xr1_INF = 1./(1.+exp((-26.-u[i, j, k])/7.))
        axr1 = 450./(1.+exp((-45.-u[i, j, k])/10.))
        bxr1 = 6./(1.+exp((u[i, j, k]-(-30.))/11.5))
        TAU_Xr1 = axr1*bxr1
        Xr2_INF = 1./(1.+exp((u[i, j, k]-(-88.))/24.))
        axr2 = 3./(1.+exp((-60.-u[i, j, k])/20.))
        bxr2 = 1.12/(1.+exp((u[i, j, k]-60.)/20.))
        TAU_Xr2 = axr2*bxr2

        Xs_INF = 1./(1.+exp((-5.-u[i, j, k])/14.))
        Axs = (1400./(sqrt(1.+exp((5.-u[i, j, k])/6))))
        Bxs = (1./(1.+exp((u[i, j, k]-35.)/15.)))
        TAU_Xs = Axs*Bxs+80

        R_INF = 0
        S_INF = 0
        TAU_R = 0
        TAU_S = 0

        R_INF = 1./(1.+exp((20-u[i, j, k])/6.))
        S_INF = 1./(1.+exp((u[i, j, k]+20)/5.))
        TAU_R = 9.5*exp(-(u[i, j, k]+40.)*(u[i, j, k]+40.)/1800.)+0.8
        TAU_S = 85.*exp(-(u[i, j, k]+45.)*(u[i, j, k]+45.) /
                        320.)+5./(1.+exp((u[i, j, k]-20.)/5.))+3.

        D_INF = 1./(1.+exp((-8-u[i, j, k])/7.5))
        Ad = 1.4/(1.+exp((-35-u[i, j, k])/13))+0.25
        Bd = 1.4/(1.+exp((u[i, j, k]+5)/5))
        Cd = 1./(1.+exp((50-u[i, j, k])/20))
        TAU_D = Ad*Bd+Cd
        F_INF = 1./(1.+exp((u[i, j, k]+20)/7))
        Af = 1102.5*exp(-(u[i, j, k]+27)*(u[i, j, k]+27)/225)
        Bf = 200./(1+exp((13-u[i, j, k])/10.))
        Cf = (180./(1+exp((u[i, j, k]+30)/10)))+20
        TAU_F = Af+Bf+Cf
        F2_INF = 0.67/(1.+exp((u[i, j, k]+35)/7))+0.33
        Af2 = 600*exp(-(u[i, j, k]+25)*(u[i, j, k]+25)/170)
        Bf2 = 31/(1.+exp((25-u[i, j, k])/10))
        Cf2 = 16/(1.+exp((u[i, j, k]+30)/10))
        TAU_F2 = Af2+Bf2+Cf2
        FCaSS_INF = 0.6/(1+(CaSS[i, j, k]/0.05)*(CaSS[i, j, k]/0.05))+0.4
        TAU_FCaSS = 80./(1+(CaSS[i, j, k]/0.05)*(CaSS[i, j, k]/0.05))+2.

        # Update gates
        M_[i, j, k] = M_INF-(M_INF-M_[i, j, k])*exp(-dt/TAU_M)
        H_[i, j, k] = H_INF-(H_INF-H_[i, j, k])*exp(-dt/TAU_H)
        J_[i, j, k] = J_INF-(J_INF-J_[i, j, k])*exp(-dt/TAU_J)
        Xr1[i, j, k] = Xr1_INF-(Xr1_INF-Xr1[i, j, k])*exp(-dt/TAU_Xr1)
        Xr2[i, j, k] = Xr2_INF-(Xr2_INF-Xr2[i, j, k])*exp(-dt/TAU_Xr2)
        Xs[i, j, k] = Xs_INF-(Xs_INF-Xs[i, j, k])*exp(-dt/TAU_Xs)
        S_[i, j, k] = S_INF-(S_INF-S_[i, j, k])*exp(-dt/TAU_S)
        R_[i, j, k] = R_INF-(R_INF-R_[i, j, k])*exp(-dt/TAU_R)
        D_[i, j, k] = D_INF-(D_INF-D_[i, j, k])*exp(-dt/TAU_D)
        F_[i, j, k] = F_INF-(F_INF-F_[i, j, k])*exp(-dt/TAU_F)
        F2_[i, j, k] = F2_INF-(F2_INF-F2_[i, j, k])*exp(-dt/TAU_F2)
        FCass[i, j, k] = FCaSS_INF-(FCaSS_INF-FCass[i, j, k])*exp(-dt/TAU_FCaSS)


class TP06Kernels3D:
    def __init__(self):
        pass

    @staticmethod
    def get_diffuse_kernel(shape):
        if shape[-1] == 7:
            return diffuse_kernel_3d_iso
        if shape[-1] == 19:
            return diffuse_kernel_3d_aniso
        else:
            raise IncorrectWeightsShapeError(shape, 7, 19)

    @staticmethod
    def get_ionic_kernel():
        return ionic_kernel_3d
