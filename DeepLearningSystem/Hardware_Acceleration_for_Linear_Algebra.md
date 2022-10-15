Algorithms and Implementation Hardware Acceleration

---

# Outline
- General acceleration techniques
  - big models, datasets
- Case study: matrix multiplication

---

# General acceleration techniques
## Layers in machine learning frameworks
```
  ML_models x -> kernel -> h(x)
  computational graph
  tensor linear algebra libraries (create a concrete and dimensional arrays, matrix multiplication)
  targets: GPU, CPU, TPU
```

## Vectorization
- Adding two arrays of length 256
```
for i in range(256):
  C[i] = A[i] + B[i]
```
- Additional requirements: memory (A, B, C) needs to be aligned to 128 bits
```
void vecadd(float* A, float *B, float* C) {
  for (int i = 0; i < 64; ++i) {
    float4 a = load_float4(A + i*4);
    float4 b = load_float4(B + i*4);
    float4 c = add_float4(a, b);
    store_float4(C + i* 4, c);
  }
}
```

## Data layout and strides

- Question: how to store a matrix in memory
  - Row major:
    - A[i, j] => Adata[i * A.shape[1] + j]
  - Column major:
    - A[i, j] => Adata[j * A.shape[0] + i]
  - Strides format:
    - A[i, j] => Adata[i * A.strides[0] + j * A.strides[1]]
      - A.strides[0] = A.shape[1], A.strides[1] = 1: Row major 
      - A.strides[0] = 1, A.strides[0] = 1: Column major 
- example
  ```
    A[2,3] = [[1,2,3],
              [4,5,6]]
  ```
  - Row major [1,2,3,4,5,6], A[i, j] = Adata[i*3 + j]
  - Column major [1,4,2,5,3,6], A[i, j] = Adata[j*2 + i]

## Discussion about strides
- Advantages: can perform transformation/slicing in zero copy way
  - Slice: change the begin offset and shape
  - Transpose: swap the strides
  - Broadcast: insert a stride equals 0
- Disadvantages: memory access becomes not continuous
   - Makes vectorization harder
   - Many linear algebra operations may require compact the array first

## Parallelization
- Executes the computation on multiple threads
```
void vecadd(float* A, float *B, float* C) {
  #pragma omp parallel for
  for (int i = 0; i < 64; ++i) {
    float4 a = load_float4(A + i*4);
    float4 b = load_float4(B + i*4);
    float4 c = add_float4(a, b);
    store_float4(C * 4, c);
  }
}
```

---

# Case study: matrix multiplication
## Vanilla matrix multiplication
- Compute C = dot(A, B.T)
- O(n^3)
```
float A[n][n], B[n][n], C[n][n];
for (int i = 0; i < n; ++i)
  for (int j = 0; j < n; ++j) {
    C[i][j] = 0;
    for (int k = 0; k < n; ++k) {
      C[i][j] += A[i][k] * B[j][k];
    }
  }
}
```

## Memory hierarchy on modern CPUs

- Source: Latency numbers every programmer should know

| memory | Latency | slow_down |
| --- | --- | --- |
| CPU thread | |
| registers | |
| L1 cache | 0.5ns |
| L2 cache | 7ns | 14xL1 cache |
| DRAM | 200ns | 20xL2 cache, 200xL1 cache |


## Architecture aware analysis

```
dram float A[n][n], B[n][n], C[n][n];
for (int i = 0; i < n; ++i) {
  for (int j = 0; j < n; ++j) {
    register float c = 0;
    for (int k = 0; k < n; ++k) {
      register float a = A[i][k];
      register float b = B[j][k];
      c += a * b;
    }
    C[i][j] = c;
  }
}
```
| tensor | from | to | the total number |
| --- | --- | --- | --- |
| A | dram | register | n^3 |
| B | dram | register | n^3 |
| a | register | register | 1 |
| b | register | register | 1 |
| c | register | register | 1 |

- Load cost = 2 * dram_speed * n^3
- Register cost = 1 * 3 = 3

## Register tiled matrix multiplication
- v1 = m
- v2 = n
- v3 = k

![](./pictures/../register_tiled_matrix_multiplication.png)


```
dram float A[n/v1][n/v3][v1][v3];
dram float B[n/v2][n/v3][v2][v3];
dram float C[n/v1][n/v2][v1][v2];

for (int i = 0; i < n/v1; ++i) {
  for (int j = 0; j < n/v2; ++j) {
    register float c[v1][v2] = 0;
    for (int k = 0; k < n/v3; ++k) {
      register float a[v1][v3] = A[i][k];
      register float b[v2][v3] = B[j][k];
      c += dot(a, b.T);
    }
    C[i][j] = c;
  }
}
```

| tensor | from | to | the total number |
| --- | --- | --- | --- |
| A | dram | register | n^3 / v2 |
| B | dram | register | n^3 / v1 |
| a | register | register | v1*v3 |
| b | register | register | v2*v3 |
| c | register | register | v1*v2 |

- Load cost = dram_speed * (n^3/v2 + n^3/v1)
  - a[v1][v3] = A[i][k]
    - outside iterations = (n/v1)*(n/v2)*(n/v3)*v1*v3 = n^3/v2
  - b[v2][v3] = B[j][k]
    - outside iterations = (n/v1)*(n/v2)*(n/v3)*v2*v3 = n^3/v1
- Register cost = 1 * (v1*v3 + v2*v3 + v1*v2)

## Cache line aware tiling
- shared memory
![](./pictures/../cache_line_aware_tiling.png)
```
dram float A[n/b1][b1][n];
dram float B[n/b2][b2][n];
dram float C[n/b1][n/b2][b1][b2];
for (int i = 0; i < n/b1; ++i) {
  L1Cache float a[b1][n] = A[i];
  for (int j = 0; j < n/b2; ++j) {
    L1Cache b[b2][n] = B[j];
    // Sub-procedure, can apply register tiling here
    C[i][j] = dot(a, b.T);
  }
}
```
- a[b1][n] = A[i]: (n/b1)*b1*n = n^2
- b[b2][n] = B[j]: (n/b1)*(n/b2)*b1*n = n^3 / b1
| tensor | from | to | the total number |
| --- | --- | --- | --- |
| A | dram | L1 | n^2 |
| B | dram | L1 | n^3 / b1 |

Constraints:
- a[b1][n] + b[b2][n] = b1 * n + b2 * n < L1 cache size
- To still apply register blocking on dot
  - b1 % v1 == 0
  - b2 % v2 == 0

## Putting it together
- m, lm, rm
- n, ln, rn
- k, k,  k
```
dram float A[n/b1][b1/v1][n][v1];
dram float B[n/b2][b2/v2][n][v2];
for (int i = 0; i < n/b1; ++i) {
  L1Cache float a[b1/v1][n][v1] = A[i];
  for (int j = 0; j < n/b2; ++j) {
    L1Cache b[b2/v2][n][v2] = B[j];
    for (int x = 0; x < b1/v1; ++x)
      for (int y = 0; y < b2/v2; ++y) {
        register float c[v1][v2] = 0;
        for (int k = 0; k < n; ++k) {
          register float ar[v1] = a[x][k][:];
          register float br[v2] = b[y][k][:];
          C += dot(ar, br.T)
        }
      }
    }
  }
}
```
load cost = L1speed * (n^3/v2 + n^3/v1) + dramspeed * (n^2 + n^3/b1)
- ar[v1] = a[x][k][:]: (n/b1)*(n/b2)*(b1/v1)*(b2/v2)*n*v1 = n^3/v2
- br[v2] = b[y][k][:]: (n/b1)*(n/b2)*(b1/v1)*(b2/v2)*n*v2 = n^3/v1
- a[b1/v1][n][v1] = A[i]: (n/b1)*b1/v1*n*v1 = n^2
- b[b2/v2][n][v2] = B[j]: (n/b1)*(n/b2)*b2/v2*n*v2 = n^3/b1


## Key insight: memory load reuse

```
dram float A[n/v1][n/v3][v1][v3];
dram float B[n/v2][n/v3][v2][v3];
dram float C[n/v1][n/v2][v1][v2];
for (int i = 0; i < n/v1; ++i) {
  for (int j = 0; j < n/v2; ++j) {
    register float c[v1][v2] = 0;
    for (int k = 0; k < n/v3; ++k) {
      register float a[v1][v3] = A[i][k];
      register float b[v2][v3] = B[j][k];
      c += dot(a, b.T);
    }
    C[i][j] = c;
  }
}
```

| tensor | reused times |
| --- | --- |
| a | v2 |
| b | v1 |

| tensor | from | to | time cost |
| --- | --- | --- | --- |
| A | dram | register | n^3 / v2 |
| B | dram | register | n^3 / v1 |

## Common reuse patterns

```
float A[n][n];
float B[n][n];
float C[n][n];
C[i][j] = sum(A[i][k] * B[j][k], axis=k)
```
- Access of A is independent of j, tile the j dimension by v enables reuse of A for v times.
- Access of B is independent of i, tile the i dimension by v enables reuse of B for v times.
  
## Discuss: possible reuse pattern in convolution
```
float Input[n][ci][h][w];
float Weight[co][ci][K][K];
float Output[n][co][h][w];
Output[b][co][y][x] = sum(Input[b][k][y+ry][x+rx] * Weight[co][k][ry][rx], axis=[k, ry, rx])
```

---

# Hardware Acceleration Implementation
## env
- Python 3.8.10
- build
  ```
  git clone https://github.com/dlsyscourse/lecture14
  cd lecture14/
  python3 -m pip install pybind11
  mkdir build
  cd build/
  cmake ..
  make
  ```
- log
  ```
  $ cmake ..
  -- The C compiler identification is GNU 7.5.0
  -- The CXX compiler identification is GNU 7.5.0
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Check for working C compiler: /usr/bin/cc - skipped
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Check for working CXX compiler: /usr/bin/c++ - skipped
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Found Python: /usr/bin/python3.8 (found version "3.8.10") found components: Development Interpreter Development.Module Development.Embed 
  -- Performing Test HAS_FLTO
  -- Performing Test HAS_FLTO - Failed
  -- Found pybind11: /home/xi.wu/.local/lib/python3.8/site-packages/pybind11/include (found version "2.10.0")
  -- Looking for pthread.h
  -- Looking for pthread.h - found
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
  -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
  -- Looking for pthread_create in pthreads
  -- Looking for pthread_create in pthreads - not found
  -- Looking for pthread_create in pthread
  -- Looking for pthread_create in pthread - found
  -- Found Threads: TRUE  
  -- Found CUDA: /usr/local/cuda (found version "10.2") 
  -- Found cuda, building cuda backend
  Sat Oct 15 03:05:40 2022       
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 470.103.01   Driver Version: 470.103.01   CUDA Version: 11.4     |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |                               |                      |               MIG M. |
  |===============================+======================+======================|
  |   0  NVIDIA GeForce ...  Off  | 00000000:3B:00.0 Off |                  N/A |
  | 18%   26C    P0    54W / 250W |      0MiB / 11178MiB |      1%      Default |
  |                               |                      |                  N/A |
  +-------------------------------+----------------------+----------------------+
                                                                                
  +-----------------------------------------------------------------------------+
  | Processes:                                                                  |
  |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
  |        ID   ID                                                   Usage      |
  |=============================================================================|
  |  No running processes found                                                 |
  +-----------------------------------------------------------------------------+
  -- Autodetected CUDA architecture(s):  6.1
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /home/xi.wu/git/lecture14/build
  xi.wu@cudaserver:~/git/lecture14/build$ make
  [ 25%] Building NVCC (Device) object CMakeFiles/ndarray_backend_cuda.dir/src/ndarray_backend_cuda_generated_ndarray_backend_cuda.cu.o
  /home/xi.wu/git/lecture14/src/ndarray_backend_cuda.cu(99): warning: variable "gid" was declared but never referenced

  Scanning dependencies of target ndarray_backend_cuda
  [ 50%] Linking CXX shared module ../python/needle/backend_ndarray/ndarray_backend_cuda.cpython-38-x86_64-linux-gnu.so
  [ 50%] Built target ndarray_backend_cuda
  Scanning dependencies of target ndarray_backend_cpu
  [ 75%] Building CXX object CMakeFiles/ndarray_backend_cpu.dir/src/ndarray_backend_cpu.cc.o
  [100%] Linking CXX shared module ../python/needle/backend_ndarray/ndarray_backend_cpu.cpython-38-x86_64-linux-gnu.so
  [100%] Built target ndarray_backend_cpu
  ```
## cmake
```
指定cmake版本
cmake_minimum_required(VERSION 3.2)
指定项目的名称，一般和项目的文件夹名称对应
project(needle C CXX)

# 执行一个或者多个外部命令
# find correct version of Python
execute_process(COMMAND python3-config --prefix OUTPUT_VARIABLE Python_ROOT_DIR)
获取整个依赖包的头文件包含路径、库路径、库名字、版本号等
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir RESULT_VARIABLE __pybind_exit_code OUTPUT_VARIABLE __pybind_path OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${__pybind_path})

if(NOT MSVC)
  设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，但是执行的时候会出现各种问题
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

# 获取整个依赖包的头文件包含路径、库路径、库名字、版本号等
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
# 添加新element到list中
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
# 指定从一组源文件src/ndarray_backend_cpu.cc编译出一个库文件且命名为ndarray_backend_cpu.cc.o
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
# 链接导入库，按照header_file + .lib + .dll方式隐式调用动态库的.lib库
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
# Sets the correct extension for a target(ndarray_backend_cpu). You can use these targets to build complex applications.
pybind11_extension(ndarray_backend_cpu)
# Strips a target (uses CMAKE_STRIP after the target is built)
pybind11_strip(ndarray_backend_cpu)

# The syntax for the command is to list all the targets you want to change, and then provide the values you want to set next.
# directly output to ffi folder
set_target_properties(ndarray_backend_cpu PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # Sets one property on zero or more objects of a scope. The first argument determines the scope in which the property is set.
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()


####################
### CUDA BACKEND ###
####################
# 获取整个依赖包的头文件包含路径、库路径、库名字、版本号等
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")
  # 获取整个依赖包的头文件包含路径、库路径、库名字、版本号等
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  # 添加新element LINKER_LIBS 到list CUDA_CUDART_LIBRARY中
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # 执行一个或者多个外部命令
  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  # 生成静态库
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})
  # 链接导入库，按照header_file + .lib + .dll方式隐式调用动态库的.lib库
  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  # Sets the correct extension for a target(ndarray_backend_cpu). You can use these targets to build complex applications.  
  pybind11_extension(ndarray_backend_cuda)
  # Strips a target (uses CMAKE_STRIP after the target is built)
  pybind11_strip(ndarray_backend_cuda)

  # The syntax for the command is to list all the targets you want to change, and then provide the values you want to set next.
  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()
```

