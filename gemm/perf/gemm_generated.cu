
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void gemm_kernel0(float* __restrict__ LHS, float* __restrict__ RHS, float* __restrict__ OUT) {
  float OUT_local[16];
  __shared__ float LHS_shared[4064];
  __shared__ float RHS_shared[4000];
  float LHS_shared_local[2];
  float RHS_shared_local[1];
  for (int i_c = 0; i_c < 2; ++i_c) {
    for (int j_c = 0; j_c < 4; ++j_c) {
      OUT_local[(((i_c * 4) + j_c))] = 0.000000e+00f;
      OUT_local[((((i_c * 4) + j_c) + 8))] = 0.000000e+00f;
      for (int k_outer = 0; k_outer < 32; ++k_outer) {
        __syncthreads();
        for (int ax0_inner = 0; ax0_inner < 4; ++ax0_inner) {
          if (((((int)threadIdx.y) * 4) + ax0_inner) < 127) {
            if (((((((int)blockIdx.x) * 128) + (((int)threadIdx.y) * 4)) + ax0_inner) + i_c) < 1024) {
              LHS_shared[((((((int)threadIdx.y) * 128) + (ax0_inner * 32)) + ((int)threadIdx.x)))] = LHS[(((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.y) * 4096)) + (ax0_inner * 1024)) + (i_c * 1024)) + (k_outer * 32)) + ((int)threadIdx.x)))];
            }
          }
        }
        for (int ax1_inner = 0; ax1_inner < 4; ++ax1_inner) {
          if (((((int)threadIdx.x) * 4) + ax1_inner) < 125) {
            if (((((((int)blockIdx.y) * 128) + (((int)threadIdx.x) * 4)) + ax1_inner) + j_c) < 1024) {
              RHS_shared[((((((int)threadIdx.y) * 125) + (((int)threadIdx.x) * 4)) + ax1_inner))] = RHS[(((((((k_outer * 32768) + (((int)threadIdx.y) * 1024)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.x) * 4)) + ax1_inner) + j_c))];
            }
          }
        }
        __syncthreads();
        for (int k_inner_outer = 0; k_inner_outer < 32; ++k_inner_outer) {
          LHS_shared_local[(0)] = LHS_shared[(((((int)threadIdx.x) * 128) + k_inner_outer))];
          LHS_shared_local[(1)] = LHS_shared[((((((int)threadIdx.x) * 128) + k_inner_outer) + 64))];
          RHS_shared_local[(0)] = RHS_shared[(((k_inner_outer * 125) + (((int)threadIdx.y) * 4)))];
          OUT_local[(((i_c * 4) + j_c))] = (OUT_local[(((i_c * 4) + j_c))] + (LHS_shared_local[(0)] * RHS_shared_local[(0)]));
          OUT_local[((((i_c * 4) + j_c) + 8))] = (OUT_local[((((i_c * 4) + j_c) + 8))] + (LHS_shared_local[(1)] * RHS_shared_local[(0)]));
        }
      }
    }
  }
  for (int i_inner_inner_inner = 0; i_inner_inner_inner < 2; ++i_inner_inner_inner) {
    for (int j_inner_inner_outer = 0; j_inner_inner_outer < 2; ++j_inner_inner_outer) {
      for (int j_inner_inner_inner = 0; j_inner_inner_inner < 2; ++j_inner_inner_inner) {
        OUT[((((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 4)) + (j_inner_inner_outer * 2)) + j_inner_inner_inner))] = OUT_local[((((i_inner_inner_inner * 4) + (j_inner_inner_outer * 2)) + j_inner_inner_inner))];
        OUT[(((((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 4096)) + (i_inner_inner_inner * 1024)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 4)) + (j_inner_inner_outer * 2)) + j_inner_inner_inner) + 2048))] = OUT_local[(((((i_inner_inner_inner * 4) + (j_inner_inner_outer * 2)) + j_inner_inner_inner) + 8))];
      }
    }
  }
}

