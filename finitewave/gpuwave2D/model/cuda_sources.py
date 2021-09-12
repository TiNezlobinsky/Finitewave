
class Kernels:
    @staticmethod
    def get_stim_volt_kernel():
        stim_volt_kernel = """
            __global__ void stim_volt(float* u, float* stim_mask, float value,
                                      const int size_i, const int size_j) {
                const int j = blockIdx.x * blockDim.x + threadIdx.x ;
                const int i = blockIdx.y * blockDim.y + threadIdx.y ;

                if (i < size_i && j < size_j) {
                    int P = i * size_j + j;
                    u[P] = u[P] * (1.0f - stim_mask[P]) + value * stim_mask[P];
                }
            }
        """

        return stim_volt_kernel

    @staticmethod
    def get_stim_curr_kernel():
        stim_curr_kernel = """
            __global__ void stim_curr(float* u, float* stim_mask, float value,
                                      const int size_i, const int size_j) {
                // notation corresponds to CPU matrixes
                const int j = blockIdx.x * blockDim.x + threadIdx.x ;
                const int i = blockIdx.y * blockDim.y + threadIdx.y ;

                if (i < size_i && j < size_j) {
                    int P = i * size_j + j;
                    u[P] += value * stim_mask[P];
                }
            }
        """
        return stim_curr_kernel

    @staticmethod
    def get_act_time_kernel():
        act_time_kernel = """
            __global__ void act_time(float* act_t, float* u,
                                           float threshold, float t,
                                           const int size_i, const int size_j) {
                const int j = blockIdx.x * blockDim.x + threadIdx.x ;
                const int i = blockIdx.y * blockDim.y + threadIdx.y ;

                if ( i < size_i && j < size_j) {
                    int P = i * size_j + j;
                    if (u[P] > threshold && act_t[P] < 0.0f) {
                        act_t[P] = t;
                    }
                }
            }
        """
        return act_time_kernel

    @staticmethod
    def get_diff_kernel():
        diff_kernel = """
            __global__ void diffusion(float *u_new, float *u, float *c,
                                      float *w0, float *w14, float *w2,
                                      float *w3, float *w5, float *w6,
                                      float *w8, const float dt,
                                      const float dr, const int size_i,
                                      const int size_j){

                const int j = blockIdx.x * blockDim.x + threadIdx.x ;
                const int i = blockIdx.y * blockDim.y + threadIdx.y ;

                if (i > 0 && i < size_i - 1 && j > 0 && j < size_j - 1) {
                    int O = (i+1) * size_j + (j-1);
                    int F = (i+1) * size_j + (j+1);
                    int P =     i * size_j + j;
                    int K = (i-1) * size_j + (j-1);
                    int L = (i-1) * size_j + (j+1);

                    int sj = threadIdx.x + 1;
                    int si = threadIdx.y + 1;

                    __shared__ float u_sh[10][34];

                    u_sh[si-1][sj-1] = u[K];
                    __syncthreads();
                    u_sh[si-1][sj+1] = u[L];
                    __syncthreads();
                    u_sh[si+1][sj-1] = u[O];
                    __syncthreads();
                    u_sh[si+1][sj+1] = u[F];
                    __syncthreads();

                    u_new[P] = u[P] + dt * c[P] * (u_sh[si-1][sj] * w0[P] -
                                                   2.0f * u[P] * w14[P] +
                                                   u_sh[si+1][sj] * w2[P] +
                                                   u_sh[si][sj-1] * w3[P] +
                                                   u_sh[si][sj+1] * w5[P] +
                                                   (u_sh[si+1][sj+1] -
                                                    u_sh[si+1][sj-1]) * w6[P] -
                                                   (u_sh[si-1][sj+1] -
                                                    u_sh[si-1][sj-1]) * w8[P])
                                      / (dr * dr);
                }
            }
        """
        return diff_kernel

    @staticmethod
    def get_ap_curr_kernel():
        currents_kernel = """
            __global__ void currents(float* u_new, float* u, float *v,
                                     const float dt, const float a,
                                     const float k, const float eap,
                                     const float mu_1, const float mu_2,
                                     const int size_i, const int size_j) {
                const int j = blockIdx.x * blockDim.x + threadIdx.x ;
                const int i = blockIdx.y * blockDim.y + threadIdx.y ;

                if (i > 0 && i < size_i - 1 && j > 0 && j < size_j - 1) {
                    int P = i * size_j + j;
                    u_new[P] += dt * ( - k * u[P] * (u[P] - a) * (u[P] - 1.0f) -
                                u[P] * v[P]);

                    v[P] = v[P] - dt * (eap + (mu_1 * v[P]) / (mu_2 + u[P])) *
                                  (v[P] + k * u[P] * (u[P] - a - 1.0f));

                }
            }
        """
        return currents_kernel

    @staticmethod
    def get_tp06_curr_kernel():
        currents_kernel = """
            // EPI
            __constant__ float Gto = 0.294f;
            __constant__ float Gks = 0.392f;
            //__constant__ float Gks = 1.0f;

            // MCELL
            //__constant__ float Gto = 0.294f;
            //__constant__ float Gks = 0.098f;

            // ENDO
            //__constant__ float Gto = 0.073f;
            //__constant__ float Gks = 0.392f;

            __constant__ float Ko = 5.4f;
            __constant__ float Cao = 2.0f;
            __constant__ float Nao = 140.0f;

            __constant__ float Vc = 0.016404f;
            __constant__ float Vsr = 0.001094f;
            __constant__ float Vss = 0.00005468f;

            __constant__ float Bufc = 0.2f;
            __constant__ float Kbufc = 0.001f;
            __constant__ float Bufsr = 10.0f;
            __constant__ float Kbufsr = 0.3f;
            __constant__ float Bufss = 0.4f;
            __constant__ float Kbufss = 0.00025f;

            __constant__ float Vmaxup = 0.006375f;
            __constant__ float Kup = 0.00025f;
            __constant__ float Vrel = 0.102f; // 40.8
            __constant__ float k1_ = 0.15f;
            __constant__ float k2_ = 0.045f;
            __constant__ float k3 = 0.060f;
            __constant__ float k4 = 0.005f; //0.000015
            __constant__ float EC = 1.5f;
            __constant__ float maxsr = 2.5f;
            __constant__ float minsr = 1.0f;
            __constant__ float Vleak = 0.00036f;
            __constant__ float Vxfer = 0.0038f;

            __constant__ float R = 8314.472f;
            __constant__ float F = 96485.3415f;
            __constant__ float T = 310.0f;
            __constant__ float RTONF = 26.713760659695648f;

            __constant__ float CAPACITANCE = 0.185f;

            __constant__ float Gkr = 0.153f;

            __constant__ float pKNa = 0.03f;

            __constant__ float GK1 = 5.405f;

            __constant__ float GNa = 14.838f;

            __constant__ float GbNa = 0.00029f;

            __constant__ float KmK = 1.0f;
            __constant__ float KmNa = 40.0f;
            __constant__ float knak = 2.724f;

            __constant__ float GCaL = 0.00003980f;

            __constant__ float GbCa = 0.000592f;

            __constant__ float knaca = 1000.0f;
            __constant__ float KmNai = 87.5f;
            __constant__ float KmCa = 1.38f;
            __constant__ float ksat = 0.1f;
            __constant__ float n_ = 0.35f;

            __constant__ float GpCa = 0.1238f;
            __constant__ float KpCa = 0.0005f;

            __constant__ float GpK = 0.0146f;

            __global__ void currents(float* u_new, float* u, float* Cai,
                                     float* CaSR, float* CaSS, float* Nai,
                                     float* Ki, float* M_, float* H_, float* J_,
                                     float* Xr1, float* Xr2, float* Xs,
                                     float* R_, float* S_, float* D_, float* F_,
                                     float* F2_, float* FCass, float* RR,
                                     float* OO, const float dt,
                                     const int size_i, const int size_j){
                const int j = blockIdx.x * blockDim.x + threadIdx.x ;
                const int i = blockIdx.y * blockDim.y + threadIdx.y ;

                if (i > 0 && i < size_i - 1 && j > 0 && j < size_j - 1){
                    int P = i * size_j + j;

                    float Ek  = RTONF * (logf((Ko / Ki[P])));
                    float Ak1 = 0.1f / (1.0f + expf(0.06f * (u[P] - Ek - 200.0f)));
                    float Bk1 = (3.0f * expf(0.0002f * (u[P] - Ek + 100.0f)) +
                                 expf(0.1f * (u[P] - Ek - 10.0f))) /
                                (1.0f + expf(-0.5f * (u[P] - Ek)));
                    float IK = GK1 * Ak1 / (Ak1 + Bk1) * (u[P] - Ek) +
                               Gto * R_[P] * S_[P] * (u[P] - Ek) + // Ito
                               Gkr * sqrtf(Ko / 5.4f) * Xr1[P] * Xr2[P] * (u[P] - Ek) + //IKr
                               GpK * (1.0f / (1.0f + expf((25.0f - u[P]) / 5.98f))) * (u[P] - Ek); //IpK

                    float Ena = RTONF * (logf((Nao / Nai[P])));
                    float INa = GNa * M_[P] * M_[P] * M_[P] * H_[P] * J_[P] *
                                (u[P] - Ena) + GbNa * (u[P] - Ena); // IbNa

                    float Eks = RTONF * (logf((Ko + pKNa * Nao) / (Ki[P] + pKNa * Nai[P])));
                    float IKs = Gks * Xs[P] * Xs[P] * (u[P] - Eks);

                    float rec_iNaK = (1.0f / (1.0f + 0.1245f * expf(- 0.1f * u[P] * F / (R * T)) +
                                      0.0353f * expf( - u[P] * F / (R * T))));
                    float INaK = knak * (Ko / (Ko + KmK)) * (Nai[P] / (Nai[P] + KmNa)) * rec_iNaK;

                    Ki[P] -= dt * (IK + IKs - 2.0f*INaK) * (1.0f/(Vc*F)) * CAPACITANCE;

                    float INaCa = knaca * (1.0f / (KmNai * KmNai * KmNai + Nao * Nao * Nao)) * (1.0f / (KmCa + Cao)) *
                                  (1.0f / (1.0f + ksat * expf((n_ - 1.0f) * u[P] * F/(R * T)))) *
                                  (expf(n_ * u[P] * F/(R * T)) * Nai[P] * Nai[P] * Nai[P] * Cao -
                                  expf((n_ - 1.0f) * u[P] * F/(R * T)) * Nao * Nao * Nao * Cai[P] * 2.5f);
                    Nai[P] -= dt * (INa + 3.0f * INaK + 3.0f * INaCa) * (1.0f / (Vc * F)) * CAPACITANCE;

                    float Eca = 0.5f * RTONF * (logf((Cao / Cai[P])));
                    float IbCa = GbCa * (u[P] - Eca);

                    float Irel = Vrel * OO[P] * (CaSR[P] - CaSS[P]);

                    float kCaSR = maxsr - ((maxsr - minsr)/(1.0f + (EC / CaSR[P]) * (EC / CaSR[P])));
                    float k1 = k1_ / kCaSR;
                    float k2 = k2_ * kCaSR;

                    OO[P] = k1 * CaSS[P] * CaSS[P] * RR[P] / (k3 + k1 * CaSS[P] * CaSS[P]);
                    RR[P] += dt * k4 * (1.0f - RR[P]) - k2 * CaSS[P] * RR[P];

                    float Ileak = Vleak * (CaSR[P] - Cai[P]);
                    float Iup = Vmaxup / (1.0f + ((Kup * Kup)/(Cai[P] * Cai[P])));

                    float dCaSR = dt * (Iup - Irel - Ileak);
                    float CaCSQN = Bufsr * CaSR[P] / (CaSR[P] + Kbufsr);
                    float bjsr = Bufsr - CaCSQN - dCaSR - CaSR[P] + Kbufsr;
                    float cjsr = Kbufsr * (CaCSQN + dCaSR + CaSR[P]);
                    CaSR[P] = (sqrtf(bjsr * bjsr + 4.0f * cjsr) - bjsr) / 2.0f;

                    float Ixfer = Vxfer * (CaSS[P] - Cai[P]);

                    // if u[P] -> 15.0f then tmp value is NaN. We use approximation
                    // float tmp = (F * F / (R * T)) * (u[P] - 15.0f) / (expf(2.0f * (u[P] - 15.0f) * F/(R * T)) - 1.0f);

                    float tmp = (F * F / (R * T)) * (2.60577e-5 * u[P]*u[P]*u[P] + 0.00495933*u[P]*u[P] - 0.679556*u[P] + 22.3491);
                    float ICaL = GCaL * D_[P] * F_[P] * F2_[P] * FCass[P] * 4.0f *
                                 (0.25f * expf(2.0f * (u[P] - 15.0f) * F/(R * T)) * CaSS[P] - Cao)* tmp;
                    float TAU_FCaSS = 80.0f / (1.0f + (CaSS[P] / 0.05f) * (CaSS[P] / 0.05f)) + 2.0f;
                    float FCaSS_INF = 0.6f / (1.0f + (CaSS[P] / 0.05f) * (CaSS[P] / 0.05f)) + 0.4f;
                    FCass[P] = FCaSS_INF - (FCaSS_INF - FCass[P]) * expf(-dt / TAU_FCaSS);

                    float dCaSS = dt * (- Ixfer * (Vc / Vss) + Irel * (Vsr / Vss) + (-ICaL * (1.0f / (2.0f * Vss * F)) * CAPACITANCE));
                    float CaSSBuf = Bufss * CaSS[P] / (CaSS[P] + Kbufss);
                    float bcss = Bufss - CaSSBuf - dCaSS - CaSS[P] + Kbufss;
                    float ccss = Kbufss * (CaSSBuf + dCaSS + CaSS[P]);
                    CaSS[P] = (sqrtf(bcss * bcss + 4.0f * ccss) - bcss) / 2.0f;

                    float IpCa = GpCa * Cai[P] / (KpCa + Cai[P]);

                    float dCai = dt * (( - (IbCa + IpCa - 2.0f * INaCa) * (1.0f / (2.0f * Vc * F)) * CAPACITANCE) - (Iup - Ileak) * (Vsr / Vc) + Ixfer);
                    float CaBuf = Bufc * Cai[P] / (Cai[P] + Kbufc);
                    float bc = Bufc - CaBuf - dCai - Cai[P] + Kbufc;
                    float cc = Kbufc * (CaBuf + dCai + Cai[P]);
                    Cai[P] = (sqrtf(bc * bc + 4.0f * cc) - bc) / 2.0f;

                    u_new[P] -= dt * (IK + IKs + INa + IbCa + INaK + ICaL +
                                      INaCa + IpCa);

                    float AM = 1.0f / (1.0f + expf((-60.0f - u[P]) / 5.0f));
                    float BM = 0.1f / (1.0f + expf((u[P] + 35.0f) / 5.0f)) +
                               0.10f /(1.0f + expf((u[P] - 50.0f) / 200.0f));
                    float TAU_M = AM * BM;
                    float M_INF = 1.0f / ((1.0f + expf((-56.86f - u[P]) / 9.03f)) * (1.0f + expf((-56.86f - u[P]) / 9.03f)));
                    M_[P] = M_INF - (M_INF - M_[P]) * expf(-dt / TAU_M);

                    float axr1 = 450.0f / (1.0f + expf((-45.0f - u[P]) / 10.0f));
                    float bxr1 = 6.0f / (1.0f + expf((u[P] - (- 30.0f)) / 11.5f));
                    float TAU_Xr1 = axr1 * bxr1;
                    float Xr1_INF = 1.0f / (1.0f + expf((-26.0f - u[P]) / 7.0f));
                    Xr1[P] = Xr1_INF - (Xr1_INF - Xr1[P]) * expf(-dt / TAU_Xr1);

                    float axr2 = 3.0f / (1.0f + expf((-60.0f - u[P]) / 20.0f));
                    float bxr2 = 1.12f / (1.0f + expf((u[P] - 60.0f) / 20.0f));
                    float TAU_Xr2 = axr2 * bxr2;
                    float Xr2_INF = 1.0f / (1.0f + expf((u[P] - (-88.0f)) / 24.0f));
                    Xr2[P] = Xr2_INF - (Xr2_INF - Xr2[P]) * expf(-dt / TAU_Xr2);

                    float Axs    = (1400.0f / (sqrtf(1.0f + expf((5.0f - u[P]) / 6.0f))));
                    float Bxs    = (1.0f / (1.0f + expf((u[P] - 35.0f) / 15.0f)));
                    float TAU_Xs = Axs * Bxs + 80.0f;
                    float Xs_INF = 1.0f / (1.0f + expf((-5.0f - u[P]) / 14.0f));
                    Xs[P]  = Xs_INF - (Xs_INF - Xs[P]) * expf(-dt / TAU_Xs);

                    float TAU_R = 9.5f * expf(-(u[P] + 40.0f) * (u[P] + 40.0f) / 1800.0f) + 0.8f;
                    float R_INF = 1.0f / (1.0f + expf((20.0f - u[P]) / 6.0f));
                    R_[P]  = R_INF - (R_INF - R_[P]) * expf(-dt / TAU_R);

                    // EPI and MCELL

                    float TAU_S = 85.0f * expf(-(u[P] + 45.0f) * (u[P] + 45.0f) / 320.0f) +
                                  5.0f / (1.0f + expf((u[P] - 20.0f) / 5.0f)) + 3.0f;
                    float S_INF = 1.0f / (1.0f + expf((u[P] + 20.0f) / 5.0f));

                    // ENDO
                    /*
                    float TAU_S = 1000.0f * expf(-(u[P] + 67.0f) * (u[P] + 67.0f) / 1000.0f) + 8.0f;
                    float S_INF = 1.0f / (1.0f + expf((u[P] + 28.0f) / 5.0f));
                    */
                    S_[P]  = S_INF - (S_INF - S_[P]) * expf(-dt / TAU_S);


                    float Ad = 1.4f / (1.0f + expf((-35.0f - u[P]) / 13.0f)) + 0.25f;
                    float Bd = 1.4f / (1.0f + expf((u[P] + 5.0f) / 5.0f));
                    float Cd = 1.0f / (1.0f + expf((50.0f -u[P]) / 20.0f));
                    float TAU_D = Ad * Bd + Cd;
                    float D_INF = 1.0f / (1.0f + expf((-8.0f - u[P]) / 7.5f));
                    D_[P]  = D_INF - (D_INF - D_[P]) * expf(-dt / TAU_D);

                    float Af = 1102.5f * expf(-(u[P] + 27.0f) * (u[P] + 27.0f) / 225.0f);
                    float Bf = 200.0f / (1.0f + expf((13.0f - u[P]) / 10.0f));
                    float Cf = (180.0f / (1.0f + expf((u[P] + 30.0f) / 10.0f))) + 20.0f;
                    float TAU_F = Af + Bf + Cf;
                    float F_INF = 1.0f / (1.0f + expf((u[P] + 20.0f) / 7.0f));
                    F_[P]  = F_INF - (F_INF - F_[P]) * expf(-dt / TAU_F);

                    float Af2 = 600.0f * expf(-(u[P] + 25.0f) * (u[P] + 25.0f) / 170.0f);
                    float Bf2 = 31.0f / (1.0f + expf((25.0f - u[P]) / 10.0f));
                    float Cf2 = 16.0f / (1.0f + expf((u[P] + 30.0f) / 10.0f));
                    float TAU_F2 = Af2 + Bf2 + Cf2;
                    float F2_INF = 0.67f / (1.0f + expf((u[P] + 35.0f) / 7.0f)) + 0.33f;
                    F2_[P] = F2_INF - (F2_INF - F2_[P]) * expf(-dt / TAU_F2);

                    float AH_ = 0.0f;
                    float BH_ = 0.0f;
                    float AJ_ = 0.0f;
                    float BJ_ = 0.0f;

                    if (u[P] >= -40.0f){
                        AH_ = 0.0f;
                        BH_ = 0.77f / (0.13f * (1.0f + expf(-(u[P] + 10.66f) / 11.1f)));
                        AJ_ = 0.0f;
                        BJ_ = 0.6f * expf(0.057f * u[P])/(1.0f + expf(-0.1f * (u[P] + 32.0f)));
                    } else {
                        AH_ = 0.057f * expf(-(u[P] + 80.0f) / 6.8f);
                        BH_ = 2.7f * expf(0.079f * u[P]) + (3.1e5f) * expf(0.3485f * u[P]);
                        AJ_ = (-2.5428e4f * expf(0.2444f * u[P]) - 6.948e-6f*
                               expf(- 0.04391f * u[P])) * (u[P] + 37.78f) /
                              (1.0f + expf(0.311f * (u[P] + 79.23f)));
                        BJ_ = 0.02424f * expf(- 0.01052f * u[P]) / (1.0f + expf(-0.1378f * (u[P] + 40.14f)));
                    }

                    float TAU_H = 1.0f / (AH_ + BH_);
                    float TAU_J = 1.0f / (AJ_ + BJ_);
                    // float J_INF=H_INF;
                    float H_INF = 1.0f / ((1.0f + expf((u[P] + 71.55f) / 7.43f)) * (1.0f + expf((u[P] + 71.55f) / 7.43f)));
                    H_[P] = H_INF - (H_INF - H_[P]) * expf(-dt / TAU_H);
                    J_[P] = H_INF - (H_INF - J_[P]) * expf(-dt / TAU_J);
                }
            }
        """
        return currents_kernel
