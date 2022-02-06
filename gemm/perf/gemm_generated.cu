
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
  OUT[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))] = 0.000000e+00f;
  for (int k = 0; k < 64; ++k) {
    OUT[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))] = (OUT[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)))] + (LHS[(((((int)blockIdx.x) * 64) + k))] * RHS[(((k * 32) + ((int)threadIdx.x)))]));
  }
}

