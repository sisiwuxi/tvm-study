
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
extern "C" __global__ void default_function_kernel0(float* __restrict__ OUT, float* __restrict__ LHS, float* __restrict__ RHS) {
  for (int j_inner = 0; j_inner < 2; ++j_inner) {
    OUT[((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 2)) + j_inner))] = 0.000000e+00f;
    for (int k = 0; k < 2048; ++k) {
      OUT[((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 2)) + j_inner))] = (OUT[((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 2)) + j_inner))] + (LHS[(((((int)blockIdx.x) * 2048) + k))] * RHS[((((k * 2048) + (((int)threadIdx.x) * 2)) + j_inner))]));
    }
  }
}

