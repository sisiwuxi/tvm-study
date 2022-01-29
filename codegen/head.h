/*
https://tvm.apache.org/docs/dev/relay_bring_your_own_codegen.html?highlight=codegen
*/

#define CSOURCE_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)         \
    extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {             \
            out[i] = a[i] p_OP_ b[i];                       \
        }                                                   \
    }

#define CSOURCE_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
    extern "C" void p_ID_(float* a, float* b, float* out) {   \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
            for (int64_t j = 0; j < p_DIM2_; ++j) {           \
                int64_t k = i * p_DIM2_ + j;                  \
                out[k] = a[k] p_OP_ b[k];                     \
            }                                                 \
        }                                                     \
    }
