diffuse_kernel_2d_iso = """
    __global__ void diffuse_kernel_2d_iso(float* u_new, float* u, float* w0,
                                          float* w1, float* w2, float* w3,
                                          float* w4, float* mesh){
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

            u_new[P] = (u_sh[si-1][sj] * w0[P] + u_sh[si][sj-1] * w1[P] +
                        u_sh[si][sj] * w2[P] + u_sh[si][sj+1] * w3[P] +
                        u_sh[si+1][sj] * w4[P])
        }
    """


diffuse_kernel_2d_aniso = """
    __global__ void diffuse_kernel_2d_iso(float* u_new, float* u, float* w0,
                                          float* w1, float* w2, float* w3,
                                          float* w4, float* w5, float* w6,
                                          float* w7, float* w8, float* mesh){
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

            u_new[P] = (u_sh[si-1][sj-1] * w0[P] + u_sh[si-1][sj] * w1[P] +
                        u_sh[si-1][sj+1] * w2[P] + u_sh[si][sj-1] * w3[P] +
                        u_sh[si][sj] * w4[P] + u_sh[si][sj+1] * w5[P] +
                        u_sh[si+1][sj-1] * w6[P] + u_sh[si+1][sj] * w7[P] +
                        u_sh[si+1][sj+1] * w8[P])
        }
    """
