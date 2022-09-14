Algorithms and Implementation Hardware Acceleration

---

# Outline
- General acceleration techniques
- Case study: matrix multiplication

---

# General acceleration techniques
## Layers in machine learning frameworks
```
  ML_models x -> kernel -> h(x)
  computational graph
  tensor linear algebra libraries
  targets: GPU, CPU, TPU
```

## Vectorization
- Adding two arrays of length 256
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
| tensor | from | to | time cost |
| --- | --- | --- | --- |
| A | dram | register | n^3 |
| B | dram | register | n^3 |
| A | register | register | 1 |
| B | register | register | 1 |
| C | register | register | 1 |

- Load cost = 2 * dram_speed * n^3
- Register cost = 3

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

| tensor | from | to | time cost |
| --- | --- | --- | --- |
| A | dram | register | n^3 / v2 |
| B | dram | register | n^3 / v1 |
| A | register | register | v1*v3 |
| B | register | register | v2*v3 |
| C | register | register | v1*v2 |

- Load cost = dram_speed * (n^3/v2 + n^3/v1)
- Register cost = v1*v3 + v2*v3 + v1*v2

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

| tensor | from | to | time cost |
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
| --- | --- | --- | --- |
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

