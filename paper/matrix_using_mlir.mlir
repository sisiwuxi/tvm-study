// High performance GPU code generation for matrix-matrix multiolication using MLIR: some early results

// ================================================ //
// Native affine matmul
// ================================================ //
affine.for %i = 0 to %M {
  affine.for %j = 0 to %N {
    affine.for %k = 0 to %K {
      %a = affine.load %A[%i, %k] : memref<8192x8192xf16>
      %b = affine.load %B[%k, %j] : memref<8192x8192xf16>
      %c = affine.load %C[%i, %j] : memref<8192x8192xf32>
      %aq = fpext %a : f16 to f32
      %bq = fpext %b : f16 to f32
      %q = mulf %aq, %bq : f32
      %co = addf %c, %q : f32
      affine.store %co, %C[%i, %j] : memref<8192x8192xf32>
    }
  }
}

// ================================================ //
// Tiled and padded affine matmul with WMMA ops
// ================================================ //
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 64)>
#map2 = affine_map<(d0) -> (d0 + 128)>
module {
  // Shared memory buffers for A and B.
  memref.global "private" @b_smem_global : memref<64x136xf16, 3>
  memref.global "private" @a_smem_global : memref<128x72xf16, 3>
  func @main() {
    ...
    affine.for %i = 0 to 8192 step 128 {
      affine.for %j = 0 to 8192 step 128 {
        // References to shared memory buffers.
        %b_smem = memref.get_global @b_smem_global : memref<64x136xf16, 3>
        %a_smem = memref.get_global @a_smem_global : memref<128x72xf16, 3>
        // Main k-loop.
        affine.for %k = 0 to 8192 step 64 {
          // Copy loop for B.
          affine.for %copykk = #map0(%k) to #map1(%k) {
            affine.for %copyjj = #map0(%j) to #map2(%j) {
              %11 = affine.load %B[%copykk, %copyjj] : memref<8192x8192xf16>
              affine.store %11, %b_smem[%copykk - %k, %copyjj - %j] : memref<64x136xf16, 3>
            }
          }
          // Copy loop for A.
          affine.for %copyii = #map0(%i) to #map2(%i) {
            affine.for %copykk = #map0(%k) to #map1(%k) {
              %11 = affine.load %A[%copyii, %copykk] : memref<8192x8192xf16>
              affine.store %11, %a_smem[%copyii - %i, %copykk - %k] : memref<128x72xf16, 3>
            }
          }
          affine.for %ii = 0 to 128 step 64 {
            affine.for %jj = 0 to 128 step 32 {
              affine.for %kk = 0 to 64 step 32  {
                affine.for %kkk = 0 to 32 step 16 {
                  affine.for %iii = 0 to 64 step 16 {
                    affine.for %jjj = 0 to 32 step 16 {
                      ...
                      %a = gpu.subgroup_mma_load_matrix %a_smem[%11, %12] {leadDimension = 72 :index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                      %b = gpu.subgroup_mma_load_matrix %b_smem[%12, %14] {leadDimension = 136 :index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                      %c = gpu.subgroup_mma_load_matrix %C[%16, %17] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
                      %res = gpu.subgroup_mma_compute %a, %b, %c : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">gpu.subgroup_mma_store_matrix %res, %C[%16, %17] {leadDimension = 8192 :index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// ================================================ //
// Affine matmul after loop unrolling and invariant load-store hoisting
// ================================================ //
...
#map0 = affine_map<(d0, d1) -> (d0 + d1)>
...
// Thread block ‘i‘ loop.
affine.for %i = 0 to 8192 step 128 {
  // Thread block ‘j‘ loop.
  affine.for %j = 0 to 8192 step 128 {
    %b_smem = memref.get_global @b_smem_global : memref<64x136xf16, 3>
    %a_smem = memref.get_global @a_smem_global : memref<128x72xf16, 3>
    // Warp ‘i‘ loop.
    affine.for %ii = 0 to 128 step 64 {
      // Warp ‘j‘ loop.
      affine.for %jj = 0 to 128 step 32 {
        // Hoisted loads on C.
        %11 = affine.apply #map0(%i, %ii)
        %12 = affine.apply #map0(%j, %jj)
        %c_reg_0 = gpu.subgroup_mma_load_matrix %C[%11, %12] {leadDimension = 8192 : index} :memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        ...
        // Main ‘k‘-loop with loaded C operand as iter_args.
        %res:8 = affine.for %k = 0 to 8192 step 64 iter_args(%c_in_0 = %c_reg_0, %c_in_1 = %c_reg_1 ...) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) 
        {
          ...
          %a = gpu.subgroup_mma_load_matrix %a_smem[%ii, %c_in_0] {leadDimension = 72 : index} :memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %b = gpu.subgroup_mma_load_matrix %b_smem[%c_in_0, %jj] {leadDimension = 136 : index} :memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %c_res = gpu.subgroup_mma_compute %a, %b, %c_in_0 : !gpu.mma_matrix<16x16xf16, "AOp">,!gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          ...
          // Main ‘k‘-loop yielding the results of the current iteration.
          affine.yield %104, %107 ... : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
          ...
        }
        // Hoisted stores on C.
        gpu.subgroup_mma_store_matrix %res#0, %C[%11, %12] {leadDimension = 8192 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
        ...
      }
    }
  }
}

// ================================================ //
// WMMA affine matmul with shifted k-loop
// prefetch
// ================================================ //
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> (d0 + 128)>
#map6 = affine_map<(d0) -> (d0 + 64)>
// Peeled copy loops for iteration 0 of k-loop.
affine.for %copyk = 0 to 64 {
  affine.for %copyj = #map4(%j) to #map5(%j) {
    %35 = affine.load %B[%copyk, %copyj] : memref<8192x8192xf16>
    affine.store %35, %b_smem[%copyk, %copyj - %j] : memref<64x136xf16, 3>
  }
}
affine.for %copyi = #map4(%i) to #map5(%i) {
  affine.for %copyk = 0 to 64 {
    %35 = affine.load %A[%copyi, %copyk] : memref<8192x8192xf16>
    affine.store %35, %a_smem[%copyi - %i, %copyk] : memref<128x72xf16, 3>
  }
}
// Main k-loop.
affine.for %k = 0 to 8128 step 64 {
  // Copy loops for iteration ‘%k + 1‘ of k-loop.
  affine.for %copyk = #map6(%k) to #map5(%k) {
    affine.for %copyj = #map4(%j) to #map5(%j) {
      %36 = affine.load %B[%copyk, %copyj] : memref<8192x8192xf16>
      affine.store %36, %b_smem[%copyk - %k - 64, %copyj - %j] : memref<64x136xf16, 3>
    }
  }
  affine.for %copyi = #map4(%i) to #map5(%i) {
    affine.for %copyk = #map6(%k) to #map5(%k) {
      %36 = affine.load %A[%copyi, %copyk] : memref<8192x8192xf16>
      affine.store %36, %a_smem[%copyi - %i, %copyk - %k - 64] : memref<128x72xf16, 3>
    }
  }
  affine.for %kk= 0 to 64 step 32 {
    ...
  }
}
// Peeled compute loop for the last iteration of k-loop.
affine.for %arg4 = 8128 to 8192 step 64 {
  ...
}

// ================================================ //
// Vectorized copy loops
// ================================================ //

...
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0) -> (d0 + 128)>
#map6 = affine_map<(d0) -> (d0 + 64)>
...
// Cast operations for global memory memrefs.
%a_cast = memref.vector_cast %A : memref<8192x8192xf16> to memref<8192x1024xvector<8xf16>>
%b_cast = memref.vector_cast %B : memref<8192x8192xf16> to memref<8192x1024xvector<8xf16>>
// Cast operations for shared memory memrefs.
%b_smem_cast = memref.vector_cast %b_smem : memref<64x72xf16, 3> to memref<64x9xvector<8xf16>,3>
%a_smem_cast = memref.vector_cast %a_smem : memref<128x72xf16, 3> to memref<128x9xvector<8xf16>,3>
// Vectorized copy loops.
affine.for %copyk = #map6(%k) to #map5(%k) {
  affine.for %copyj = #map4(%j) to #map5(%j) step 8 {
    %135 = affine.load %b_cast[%copyk, %copyj floordiv 8] : memref<8192x1024xvector<8xf16>>affine.store %135, %b_smem_cast[%copyk - %k - 64, (%copyj - %j) floordiv 8] : memref<64x17xvector<8xf16>, 3>
  }
}
affine.for %copyi = #map4(%i) to #map5(%i) {
  affine.for %copyk = #map6(%k) to #map5(%k) step 8 {
    %135 = affine.load %a_cast[%copyi, %copyk floordiv 8] : memref<8192x1024xvector<8xf16>>affine.store %135, %a_smem_cast[%copyi - %i, (%copyk - %k) floordiv 8 - 8] : memref<128x9xvector<8xf16>, 3>
  }
}

// ================================================ //
// Global memory load latency hiding
// ================================================ //
gpu.launch blocks(%blockIdX, %blockIdY, %blockIdX) in (%arg6 = %c64, %arg7 = %c64, %arg8 = %c1)
threads(%threadIdX, %threadIdY, %threadIdZ) in (%arg9 = %c256, %arg10 = %c1, %arg11 = %c1)
{
  ...
  %c_reg_0 = gpu.subgroup_mma_load_matrix %C[%26, %27] {leadDimension = 8192 : index} : memref<8192x8192xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
  ...
  // Peeled copy loops for iteration 0 of k-loop.
  scf.for %copy = %c0 to %c4 step %c1 {
    ...
  }
  scf.for %copy = %c0 to %c4 step %c1 {
    ...
  }
  gpu.barrier
  // Main k-loop
  %res:8 = scf.for %k = %c0 to %c8128 step %c64 iter_args(%c_in_0 = %c_reg_0, %c_in_1 = %c_reg_1...) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp"> ...) {
    gpu.barrier
    // Global memory loads for iteration i + 1 of k-loop
    %a_next_iter_0 = memref.load %a_cast[%74, %81] : memref<8192x1024xvector<8xf16>>
    %b_next_iter_0 = memref.load %b_cast[%94, %101] : memref<8192x1024xvector<8xf16>>
    ...
    scf.for %kk = %c0 to %c64 step %c32 iter_args(%arg16 = %c_in_0, %arg17 = %c_in_1) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp"> 
    {
      ...
    }
    gpu.barrier
    // Shared memory stores for iteration i + 1 of k-loop
    memref.store %b_next_iter_0, %b_smem_cast[%51, %68] : memref<64x17xvector<8xf16>, 3>
    memref.store %a_next_iter_0, %a_smem_cast[%150, %167] : memref<128x9xvector<8xf16>, 3>
    ...
  }
  gpu.barrier
  // Peeled compute loop for iteration n-1 of k-loop.
  scf.for %arg14 = %c0 to %c64 step %c32 {
    ...
  }
  gpu.subgroup_mma_store_matrix %res#0, %C[%26, %27] {leadDimension = 8192 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<8192x8192xf32>
  ...
}