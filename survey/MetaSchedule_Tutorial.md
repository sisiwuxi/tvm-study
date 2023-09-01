# TVM Unity - MetaSchedule
- MetaSchedule: TIR -> TIR Transformation

# intrudoction
## tensor program
- tensors: n-dimentional arrays
- operators: a linear algebra operator(e.g, convolution, matrix multiplication) or a non-linear activation(e.g, relu, sigmoid)
## tensor program eaxmple-matrix matmul(TE)
- batch_matmul_nkkm
## optimization
- before 
- after, 100xfaster
- key elements in tensor program optimization
    - initial -> search space -> search algorithm -> optimal program
## how search space is defined
- different transformation lead to different tensor program outcome. search space is defined by a set of sequence of program transformation.
- correctness transformations include:
    - split
    - reorder
    - cache read/write
    - thread binding
    - vectorize
    - parallel
## parameterized transformation
- parameterization
- equivalent programs included by parameterized transformation
- different parameterization may transform the initial tensor program into very different tensor
- programs missing the process virtually unsustablable if done manually
## the missing piece...
- the transformations has to be parameterized before being aplied to the tensor program
- different parameterization may transform the initial tensor program into vastly different intermediate tensor programs, making it hard to apply even further transformations
## stochastic transformation
- transformation module
    - program transformation based on random variables
    - analysis based stochastic transformation
- random variable from sampling primitives
    - sample-perf-tile
        - how split a loop into smaller loop
    - sample-categorical
        - how we're going to make decision choices
            - how select some numbers to do thread binding
    - sample-compute-location
        - where we're gonna do some fusion
        - choose some space we're gonna apply certain optimization
        - where compute_at
## MetaSchedule
- MetaSchedule is TVM's latest generation of automated scheduling technology that could support such abstraction and explore the search space efficiently
- MetaSchedule leverages a probabilistic programming approach to generate a wide search space and introduces sampling to define the space easily
## MetaSchedule - Terminologies
- transformation module: schedule rules
- parameterization: trace decision
- transformation: schedule primitives(trace instructions)
## TVM unitt - MetaSchedule
- MetaSchedule is TVM's latest generation of automated scheduling technology that unifies all previous versions under a single program interface
    - manual schedule
    - autoTvm style template search
    - ansor style search rule
## MetaSchedule example
- TE manual schedule
    - supported as well in TIR
- autoTvm style template search
    - use sampling for knobs
- ansor style search rule
    - supported in ms schedule rules
    - easy to customize
## incorporating domain knowledge is easy
- use tensor core, instruction set
## validation MetaSchedule's search space
- can MetaSchedule cover the search space of previous searching methods?
    - can MetaSchedule deliver better or comparable results?
- can MetaSchedule do difficient search/tuning over the given search space?
- how domain knowladge can expand search space and improve final performance in MetaSchedule?
## performance overview - operator/subgraph
- is comparable or better than ansor and outperformans most poytorch
## performance overview - end to end
## performance overview
- usgae of transformation modules across both ops/subgraph and full end workloads result in performance improvements over naive schedule implementations and industry standard ML frameworks
- to summarize, MetaSchedule is bale to:
    - cover the search space defined in previous interations of automated scheduling technologies
    - deliver performance that either better or comparable to industry leading solutions by exploring a given search space
    - has the flexibility to inject domain knowledge into the search space using search rules which can be customized to fit the needs of future model architectures and devices

# MetaSchedule Tuning
## before tuning
- TE program is transformed into TensorIR program
- Relay Graphs are dispatched to multiple TensorIR programs
    ```
        TE program        Relay Graphs  
            \                  /
        createPrimFunc      TaskExtraction
            \              /
            TensorIR programs
    ```
## TE program
- tvm/python/tvm/meta_schedule/testing/te_workload.py:25
    ```
    def batch_matmul_nkkm(
        B: int,
        N: int,
        M: int,
        K: int,
        in_dtype: str = "float32",
        out_dtype: str = "float32",
    ) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
        x = te.placeholder((B, N, K), name="X", dtype=in_dtype)
        y = te.placeholder((B, K, M), name="Y", dtype=in_dtype)
        k = te.reduce_axis((0, K), name="k")
        z = te.compute(  # pylint: disable=invalid-name
            (B, N, M),
            lambda b, i, j: te.sum(
                x[b][i][k].astype(out_dtype)*y[b][k][j].astype(out_dtype),
                axis=[k],
            ),
            name="Z",
        )
        return (x, y, z)
    ```
- create_te_workload("GMM", 0)
  ```
    @T.prim_func
    def main(X: T.Buffer((1, 128, 128), "float32"), Y: T.Buffer((1, 128, 128), "float32"), Z: T.Buffer((1, 128, 128), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for b, i, j, k in T.grid(1, 128, 128, 128):
            with T.block("Z"):
                v_b, v_i, v_j, v_k = T.axis.remap("SSSR", [b, i, j, k])
                T.reads(X[v_b, v_i, v_k], Y[v_b, v_k, v_j])
                T.writes(Z[v_b, v_i, v_j])
                with T.init():
                    Z[v_b, v_i, v_j] = T.float32(0)
                Z[v_b, v_i, v_j] = Z[v_b, v_i, v_j] + X[v_b, v_i, v_k]*Y[v_b, v_k, v_j]
  ```
## TensorIR program Tuning - Overview
- search space generation
- sample candidates from search space
- candidate evaluation
- save results

## space generation
- produces a list of schedules as the search generated by the tensor program
- schdule rules are applies to each piece of TIR

## TensorIR program Tuning - Example
- for i, j, k in T.grid(4, 6, 8):
- ./src/tir/schedule/trace.cc
- ./tests/python/unittest/test_meta_schedule_post_order_apply.py
  - test_meta_schedule_post_order_apply_remove_block
    - mod = TrinityMatmul
    - schs = post_order_apply.generate_design_space(mod)
      - python/tvm/meta_schedule/testing/space_generation.py:57
      - python/tvm/meta_schedule/tune_context.py:150
      - python/tvm/meta_schedule/space_generator/space_generator.py:73
        - _ffi_api.SpaceGeneratorGenerateDesignSpace(self, mod)
        - ./src/meta_schedule/space_generator/space_generator.cc：170
          - PySpaceGeneratorNode::GenerateDesignSpace
            - f_generate_design_space(mod)
        - ./include/tvm/meta_schedule/space_generator.h
          - using FGenerateDesignSpace = SpaceGenerator::FGenerateDesignSpace;
          - using FGenerateDesignSpace = runtime::TypedPackedFunc<Array<tir::Schedule>(const IRModule&)>;
    - str(sch_trace) == correct_trace([2, 512], [2, 512], [2, 512], [2, 512])
- ./tests/python/unittest/test_meta_schedule_trace_apply.py
  - test_conv2d_int8_tensorcore
    - verify(Conv2dInt8, apply_trace, Conv2dInt8_target, "cuda", Conv2dInt8_tensorcore_scheduled)
  - class Conv2dInt8
    - NHiWiCi, CoKhKwCi, NHoWoCo
    - tir.noalias: https://llvm.org/docs/AliasAnalysis.html#MustMayNo
    - spatial, reduction
    - conv2d
    ```
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with T.block("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                T.writes(conv2d_nhwc[nn, yy, xx, ff])
                with T.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], "int32")*T.cast(p1[ff, ry, rx, rc], "int32")
    -->
        for i0 in range(N):
            for i1 in range(Ho):
                for i2 in range(Wo):
                    for i3 in range(Co):
                        nn, yy, xx, ff = i0, i1, i2, i3
                        res[nn, yy, xx, ff] = 0
                        for i4 in range(Ci):
                            for i5 in range(R):
                                for i6 in range(S):
                                    ry, rx, rc = i4, i5, i6
                                    res[nn, yy, xx, ff] += pad_temp[nn, yy + ry, xx + rx, rc]*p1s[ff, ry, rx, rc]
    ```
    - relu
    ```
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with T.block("compute"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(T_add_2[i0_2, i1_2, i2_2, i3_2])
                T.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.max(T.min(T_add_2[i0_2, i1_2, i2_2, i3_2], 255), 0)
    -->
        compute_1 = np.maximum(np.zeros_like(res), res)
    ```
    - q_multiply_shift
        - https://en.wikipedia.org/wiki/Q_(number_format)
            - Q8.8
                - integer part have 8 bits
                - fraction part have 8 bits
        - ./python/tvm/tir/op.py:2603
            - out = round(x*y*2^-s)
  - apply_trace
    - sch = Schedule(target_mod)
        - transform_layout
        - transform_block_layout
        - get_loops
        - get_block
        - reorder
        - blockize
        - annotate
            - meta_schedule.auto_tensorize
                - wmma_sync_16x16x16_s8s8s32_trans
                - wmma_store_16x16x16_s32_shared
                - wmma_load_16x16x16_s8_a_shared
                - wmma_load_16x16x16_s8_b_trans_shared
                - wmma_fill_16x16x16_s32
            - meta_schedule.auto_tensorize_init = wmma_fill_16x16x16_s32
            - warp_execution = 1
            - meta_schedule.thread_extent_low_inclusive = 32
            - meta_schedule.thread_extent_high_inclusive = 1024
            - meta_schedule.cooperative_fetch = v87
            - meta_schedule.unroll_explicit = v193
            - pragma_auto_unroll_max_step = 512
            - pragma_unroll_explicit = 1
        - sample_perfect_tile
        - split
        - fuse
        - bind
        - cache_write
        - compute_at
        - reverse_compute_at
        - sample_categorical
        - reverse_compute_inline
        - cache_read
        - compute_inline
        - storage_align
        - enter_postproc
        - vectorize
        - unannotate
            - meta_schedule.auto_tensorize_init
            - meta_schedule.auto_tensorize
        - tensorize
            - wmma_fill_16x16x16_s32
            - wmma_load_16x16x16_s8_a_shared
            - wmma_load_16x16x16_s8_b_trans_shared
            - wmma_sync_16x16x16_s8s8s32_trans
            - wmma_store_16x16x16_s32_shared
  - Conv2dInt8_tensorcore_scheduled
    - kernel.cu
    - T
        - block
        - grid
        - axis.spatial
        - reads
        - writes
        - block_attr
        - match_buffer
        - tvm_fill_fragment
        - serial
        - tvm_load_matrix_sync
        - axis.remap
        - axis.reduce
        - tvm_mma_sync
        - tvm_store_matrix_sync
    - compute[v0//3136, v0 % 3136//56, v0 % 56, v1] = T.max(T.min(T.cast(T.max(T.min(T.q_multiply_shift(T.cast(T.cast(T.max(T.min(p7[()] + T.cast(T.shift_right(T.cast(conv2d_nhwc_reindex_shared[v0, v1] - p2[0, 0, 0, v1] + p3[0, 0, 0, v1], "int64")*p4[0, 0, 0, v1] + p5[0, 0, 0, v1], p6[0, 0, 0, v1], dtype="int64"), "int32"), 255), 0), "uint8"), "int32") - p8[0], 1098990753, 31, 1, dtype="int32") + p9[v0//3136, v0 % 3136//56, v0 % 56, v1], 255), 0), "uint8"), T.uint8(255)), T.uint8(0))


    
## TensorIR program Tuning - Search Strategy
- explores the seach space by sampling desicions from the trace geenrated
- example
    - replay trace: randomly generation
    - evolutionary search: generate via evolution search, with help from cost model
        - ./src/meta_schedule/search_strategy/evolutionary_search.cc

## TensorIR program Tuning - evolutionary search
- cost model are trained with real benchmark results
- candidite 1: cost model = 0.1
- candidite 2: cost model = 0.5
- selected candidates
    - select very top quality
    - candidite 1: 0.91
    - candidite 2: 0.95
    - candidite N: 0.99

## TensorIR program Tuning - Builder & Runner
- builder builds TIR workloads into Runtime Module
- Runner gets benchmark results
- Support gets benchmark results
- Support RPCRunner to benchmark on target device

## TensorIR program Tuning - Measure Callback
- Tuning results are being stored into Database as tuning records
- Tuning results are being used to train the Cost Model
- Statistics are collected for debugging

## TensorIR program Tuning - Collect Callback
- Tuning wil return a databse for later queries
- QuerySchedule API can get the best result for a given TIR workload from the database
    - ./src/meta_schedule/database/database.cc
- GetTopK API can find more tuning records with benchmarking results available
    - ./src/meta_schedule/database/json_database.cc

# future
## tuning API
- TIR tuning: MetaSchedule Database
- TE tuning: MetaSchedule Database
- Relay tuning: MetaSchedule Database
    - Task Extraction
    - Tune Extraction Tasks
- Relay Compilation -> Runtime Module
    - Compile for TVM Backend
- ONNX/PyTorch Tuning Script
    - Frontend + Relay Tuning + Compilation + Bechmarking

## AutoTensorization
### Register in TIR Tensor Intrins
    - Example:
        - vnni in Intel AVX 512 CPU
    - steps
        - Description for pattern matching
            - ./tests/python/unittest/test_tir_schedule_analysis.py
              - get_tensorize_loop_mapping
                - dot_product_16x4_u8i8i32_desc
                - matmul_16x16x16xf16f16f16_desc
                - WMMA_SYNC_16x16x16_f16f16f16_INTRIN
              - test_get_tensorize_loop_mapping_dense_16x4
                - DenseTIRModule
                - block = s.get_block("compute")
                  ```
                    with T.block("compute"):
                        i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                        T.reads(placeholder[i, k], placeholder_1[j//16, k//4, j % 16, k % 4])
                        T.writes(compute[i, j])
                        with T.init():
                            compute[i, j] = 0
                        compute[i, j] = compute[i, j] + T.Cast("int32", placeholder[i, k])*T.Cast("int32", placeholder_1[j//16, k//4, j % 16, k % 4])
                  ```
                - info = get_tensorize_loop_mapping(s, block, dot_product_16x4_u8i8i32_desc)
              - test_get_tensorize_loop_mapping_conv2d_nchwc_16x4
                - Conv2dNCHWcTIRModule
                - block = s.get_block("conv2d_NCHWc_int8")
                    ```
                        with T.block("conv2d_NCHWc_int8", no_realize=True):
                            n = T.axis.spatial(1)
                            oc_chunk = T.axis.spatial(16)
                            oh = T.axis.spatial(56)
                            ow = T.axis.spatial(56)
                            oc_block = T.axis.spatial(16)
                            kh = T.axis.reduce(1)
                            kw = T.axis.reduce(1)
                            ic_outer = T.axis.reduce(4)
                            ic_f_inner = T.axis.reduce(4)
                            ic_s_inner = T.axis.reduce(4)
                            placeholder = T.Buffer((1, 4, 56, 56, 16), "uint8")
                            placeholder_1 = T.Buffer((16, 4, 1, 1, 4, 16, 4), "int8")
                            T.reads(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner*4 + ic_s_inner], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner])
                            conv2d_NCHWc_int8 = T.Buffer((1, 16, 56, 56, 16), "int32")
                            T.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                            with T.init():
                                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] + T.Cast("int32", placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner*4 + ic_s_inner])*T.Cast("int32", placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner])                
                    ```
                - info = get_tensorize_loop_mapping(s, block, dot_product_16x4_u8i8i32_desc)
              
        - Implementation
            - ./python/tvm/tir/tensor_intrin/cuda.py
                ```
                    WMMA_SYNC_16x16x16_f16f16f16_INTRIN = "wmma_sync_16x16x16_f16f16f16"
                    TensorIntrin.register(
                        WMMA_SYNC_16x16x16_f16f16f16_INTRIN,
                        *get_wmma_sync_intrin(16, 16, 16, "float16", "float16", False),
                    )            
                ```
            - ./python/tvm/tir/tensor_intrin/x86.py
              - dot_product_16x4_u8i8i32_desc
                ```
                    @T.prim_func
                    def dot_product_16x4_u8i8i32_desc( A: T.Buffer((4,), "uint8", offset_factor=1), B: T.Buffer((16, 4), "int8", offset_factor=1), C: T.Buffer((16,), "int32", offset_factor=1),
                    ) -> None:
                        with T.block("root"):
                            T.reads(C[0:16], A[0:4], B[0:16, 0:4])
                            T.writes(C[0:16])
                            for i in T.serial(0, 16):
                                for k in T.serial(0, 4):
                                    with T.block("update"):
                                        vi, vk = T.axis.remap("SR", [i, k])
                                        C[vi] = C[vi] + T.cast(A[vk], "int32")*T.cast(B[vi, vk], "int32")

                ```
              - dot_product_16x4_u8i8i32_vnni
                ```
                    C[T.ramp(T.int32(0), 1, 16)] = T.call_llvm_pure_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512"),
                        T.uint32(3),
                        C_i32x16,
                        T.broadcast(A_i32, 16),
                        B_i32x16,
                        dtype="int32x16",
                    )
                ```
              - dot_product_16x4_u8i8i32_avx512
                ```
                    Red = T.call_llvm_pure_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.x86.avx512.pmaddubs.w.512"),
                        T.uint32(2),
                        A_u8x64,
                        B_i8x64,
                        dtype="int16x32",
                    )

                    C[T.ramp(T.int32(0), 1, 16)] += T.call_llvm_pure_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.x86.avx512.pmaddw.d.512"),
                        T.uint32(2),
                        Red,
                        T.int16x32(1),
                        dtype="int32x16",
                    )
                ```
        - Registration
            - ./tests/python/unittest/test_meta_schedule_postproc_rewrite_tensorize.py
              - test_rewrite_tensorize_dense_dp4a
                - RewriteCooperativeFetch
                - RewriteReductionBlock
                  - decouple compute_o to compute_o_init and compute_o_update
                - RewriteTensorize
                  - match and replace T.block_attr({"meta_schedule.auto_tensorize": "dp4a"})
                  - compute_local[i, j] = compute_local[i, j] + T.Cast("int32", X_shared[i, k_o*4 + k])*T.Cast("int32", W_shared[j, k_o*4 + k])
                  - C[0] = C[0] + T.call_pure_extern("int32", "__dp4a", A[0:4], B[0:4], 0)
            - ./src/meta_schedule/schedule_rule/schedule_rule.cc
            - ./python/tvm/tir/tensor_intrin/x86.py
### Enable in intrinsic schedule rules
### full example see
    - ./tests/python/integration/test_auto_tensorize.py
      - test_dp4a_dense
        - _test_dense("int8", SCH_RULES_FOR_DP4A, POSTPROCS_FOR_DP4A, "nvidia/geforce-rtx-3070")
          - SCH_RULES_FOR_DP4A
            - _get_sch_rules_for_dp4a(DP4A_INTRIN)
              - DP4A_INTRIN
                - python/tvm/tir/tensor_intrin/dot_product_common.py
                  - C[0] = C[0] + T.cast(A[vi], "int32") * T.cast(B[vi], "int32")
                  - C[0] += T.call_pure_extern("__dp4a", A.vload([0], "int8x4"), B.vload([0], "int8x4"), T.int32(0), dtype="int32")
              - _get_sch_rules_for_dp4a
                - MultiLevelTilingWithIntrin
                - AutoInline
                - CrossThreadReduction
                - ParallelizeVectorizeUnroll
          - POSTPROCS_FOR_VNNI
            - DisallowDynamicLoop
            - RewriteParallelVectorizeUnroll
            - RewriteReductionBlock
            - RewriteTensorize
          - POSTPROCS_FOR_DP4A
            - DisallowDynamicLoop
            - RewriteCooperativeFetch
            - RewriteUnboundBlock
            - RewriteParallelVectorizeUnroll
            - RewriteReductionBlock
            - RewriteTensorize
            - VerifyGPUCode
          - tune_and_test(relay_mod, data_np, weight_np, "dense", target, sch_rules, postprocs)
            - tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts
              - tasks[0]=TuneContext
                - python/tvm/meta_schedule/tune_context.py 
                  - mod
                  - target
                  - space_generator
                    - python/tvm/meta_schedule/space_generator/space_generator.py
                  - search_strategy
                    - python/tvm/meta_schedule/search_strategy/search_strategy.py
                  - task_name
            - database=JSONDatabase
              - python/tvm/meta_schedule/database/json_database.py
              - path_workload
              - path_tuning_record
            - lib = relay.build(relay_mod, target=target, params=params)
              - tvm/python/tvm/relay/backend/executor_factory.py
              - asm = lib.lib.get_source(fmt="asm") // "s"
              - llvmir = lib.lib.get_source("ll") // "c"

- Easy Customization
    - All major classes are easy to customize from Python side
        - PyScheduleRule
        - PyBuilder
        - PyRunner
        - ...
        - Just need to override functions
        - ./tests/python/unittest/test_meta_schedule_post_order_apply.py
        ```
        @derived_object class WowSoFancyScheduleRule(PyScheduleRule):
        def _initialize_with_tune_context(self, context: "TuneContext") -> None:
            pass

        def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
            if _is_root(sch, block):
                return [sch]
            new_sch = sch.copy()
            i, j, k = new_sch.get_loops(block=block)
            i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[2, 4, 64, 2])
            j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[4, 64, 2, 2])
            k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
            new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
            return [new_sch]
        ```
    - Decouple tuning & compilation
        - tuning result is in a database
        - any database can be applied during compilation

# Logging interface for MetaSchedule
- A tale of two loggers
    - python side logging
        - using pythonlogger
        - python logging levels
    - tvm c++ side logging
        - output to screen
        - c++ side logging levels
- unified tvm py logger(via FFI)
    - unified logging level for c++/python
        - debug/info/warn
    - unified configuration on python side
        - via organic python loggers
    - rich details in logging
        - print both to console and file
        - multi-level logging
- task scheduler logger ---> console logging/notbook output
    - loggers -> work_dir/logs
- performance debugging w/MS logger - Perf Table
    - for the given subgraph, the initial IRModule before schedule, the search space(one ot multiple schedules, demonstrated with IRModule and corresponding trace), and tuning trials are available in th logging file
    - evolutionary search would print out statistics of post processors(process and filter out invalid candidates) in each step
    - number of sampled candidates and iteration numbers are also in the output
    - most importantly, performance result of each trail would be printed out, and we can check the log to find building / running issues
    - after checking the perf log, we can use tune_tir api to directly do tuning with the subgraphs' original TIR and print out intermediate results
        - check if the search space doesn't include the optimal schedule, modify schedule rules
        - checki if tuning trials number if enough to find the optimal schedule
        - if the subgraphs' performance table looks good, we can examine if the workload is correctly dispatched in the compilation logging(output console)
            - check for warning of "Cannot find workload: xxx" during compilation of relay graph
- efficiency benchmarking - MetaSchedule Profiler
    - tuning time is tracked for each tuning
    - Easy to find the most time consuming part
    - Can be used for customized benchmarking

# key takeaways
- MetaSchedule can support easy definition of stochastic search space
- Domain knowledge can be inkected into the search space of MetaSchedule
- MetaSchedule can deliver performance by efficiently exploring the search space

# reference
- https://arxiv.org/abs/2205.13603

# QA
## what is the biggest difference from ansor?
- MetaSchedule is a framework completely based on tensorIR which solid support sampling and rundown variables
- Ansor is hard to integrate new domain knowledge
## how do you handle error
- if all fail
    - which task actually failed
    - use tune TIR API to tune this task itself, reproduce the error
    - which step went wrong
    - example
        - search space is too restrictive so there's no valid candidate
        - too stricted post processors to allow
        - benchmarking system has had it's not running correctly
## if the search space have too much invalide codes that cannot actually run, how can we optimize this search space or make the running process more efficient
- search space is generated base on search rules
- tuning experience
## whether auto-scheduler logs can be used as start point for MetaSchedule?
- no, support start from MetaSchedule log
## backward compatiblility mechanism in MetaSchedule to apply tuned result from previous TVM version on newer one?
- not recently


# log
## TemporaryDirectory
- database_tuning_record.json
  - tuning record table
  - schedule_1
    ```
        [0,[[[
        ["GetBlock",[],["compute","main"],["b0"]],
        ["GetBlock",[],["root","main"],["b1"]],
        ["Annotate",["b0","\"SSSRRSRS\""],["meta_schedule.tiling_structure"],[]],
        ["GetLoops",["b0"],[],["l2","l3","l4"]],
        ["Split",["l4","None",4],[1],["l5","l6"]],
        ["Reorder",["l6"],[],[]],
        ["Blockize",["l6"],[true],["b7"]],
        ["Annotate",["b7","\"dp4a\""],
        ["meta_schedule.auto_tensorize"],[]],
        ["GetLoops",["b7"],[],["l8","l9","l10"]],
        ["SamplePerfectTile",["l8"],[5,64],["v11","v12","v13","v14","v15"]],
        ["Split",["l8","v11","v12","v13","v14","v15"],[1],["l16","l17","l18","l19","l20"]],
        ["SamplePerfectTile",["l9"],[5,64],["v21","v22","v23","v24","v25"]],
        ["Split",["l9","v21","v22","v23","v24","v25"],[1],["l26","l27","l28","l29","l30"]],
        ["SamplePerfectTile",["l10"],[3,64],["v31","v32","v33"]],
        ["Split",["l10","v31","v32","v33"],[1],["l34","l35","l36"]],
        ["Reorder",["l16","l26","l17","l27","l18","l28","l34","l35","l19","l29","l36","l20","l30"],[],[]],
        ["Fuse",["l16","l26"],[1],["l37"]],
        ["Bind",["l37"],["blockIdx.x"],[]],
        ["Fuse",["l17","l27"],[1],["l38"]],
        ["Bind",["l38"],["vthread.x"],[]],
        ["Fuse",["l18","l28"],[1],["l39"]],
        ["Bind",["l39"],["threadIdx.x"],[]],
        ["Annotate",["b7",32],["meta_schedule.thread_extent_low_inclusive"],[]],
        ["Annotate",["b7",1024],["meta_schedule.thread_extent_high_inclusive"],[]],
        ["CacheWrite",["b7",[]],[0,"local"],["b40"]],
        ["ReverseComputeAt",["b40","l39"],[1,-1],[]],
        ["CacheRead",["b7",["b7"]],[0,"shared"],["b41"]],
        ["ComputeAt",["b41","l34"],[1,-1],[]],
        ["GetLoops",["b41"],[],["l42","l43","l44","l45","l46","l47"]],
        ["Fuse",["l46","l47"],[1],["l48"]],
        ["SampleCategorical",[],[[1,2,3,4],[0.25,0.25,0.25,0.25]],["v49"]],
        ["Annotate",["b41","v49"],["meta_schedule.cooperative_fetch"],[]],
        ["CacheRead",["b7",["b7"]],[1,"shared"],["b50"]],
        ["ComputeAt",["b50","l34"],[1,-1],[]],
        ["GetLoops",["b50"],[],["l51","l52","l53","l54","l55","l56"]],
        ["Fuse",["l55","l56"],[1],["l57"]],
        ["SampleCategorical",[],[[1,2,3,4],[0.25,0.25,0.25,0.25]],["v58"]],
        ["Annotate",["b50","v58"],["meta_schedule.cooperative_fetch"],[]],
        ["SampleCategorical",[],[[0,16,64,512,1024],[0.2000000000000000111,0.2000000000000000111,0.2000000000000000111,0.2000000000000000111,0.2000000000000000111]],["v59"]],
        ["Annotate",["b1","v59"],["meta_schedule.unroll_explicit"],[]],
        ["EnterPostproc",[],[],[]],
        ["Unannotate",["b41"],["meta_schedule.cooperative_fetch"],[]],
        ["GetLoops",["b41"],[],["l60","l61","l62","l63","l64"]],
        ["Split",["l64","None",32,4],[1],["l65","l66","l67"]],
        ["Vectorize",["l67"],[],[]],
        ["Bind",["l66"],["threadIdx.x"],[]],
        ["Unannotate",["b50"],["meta_schedule.cooperative_fetch"],[]],
        ["GetLoops",["b50"],[],["l68","l69","l70","l71","l72"]],
        ["Split",["l72","None",32,4],[1],["l73","l74","l75"]],
        ["Vectorize",["l75"],[],[]],
        ["Bind",["l74"],["threadIdx.x"],[]],
        ["GetBlock",[],["root","main"],["b76"]],
        ["Unannotate",["b76"],["meta_schedule.unroll_explicit"],[]],
        ["GetChildBlocks",["b76"],[],["b77","b78","b79","b80"]],
        ["GetLoops",["b77"],[],["l81","l82","l83","l84","l85","l86","l87"]],
        ["GetLoops",["b78"],[],["l88","l89","l90","l91","l92","l93","l94"]],
        ["GetLoops",["b79"],[],["l95","l96","l97","l98","l99","l100","l101","l102","l103","l104"]],
        ["Annotate",["l95",16],["pragma_auto_unroll_max_step"],[]],
        ["Annotate",["l95",1],["pragma_unroll_explicit"],[]],
        ["GetLoops",["b80"],[],["l105","l106","l107","l108","l109"]],
        ["GetBlock",[],["compute_o","main"],["b110"]],
        ["GetLoops",["b110"],[],["l111","l112","l113","l114","l115","l116","l117","l118","l119","l120"]],
        ["DecomposeReduction",["b110","l114"],[],["b121"]],
        ["Unannotate",["b121"],["meta_schedule.auto_tensorize"],[]],
        ["Annotate",["b121","\"\""],["meta_schedule.auto_tensorize"],[]],
        ["GetBlock",[],["compute_o_update","main"],["b122"]],
        ["Unannotate",["b122"],["meta_schedule.auto_tensorize"],[]],
        ["Tensorize",["b122"],["dp4a",true],[]]
        ],
        [[9,[16,1,2,4,8]],[11,[8,1,16,1,8]],[13,[8,32,1]],[30,3],[36,3],[38,1]]],
        [10000000000],
        {"arch":"sm_86","keys":["cuda","gpu"],"kind":"cuda","max_num_threads":1024,"max_shared_memory_per_block":49152,"max_threads_per_block":1024,"registers_per_block":65536,"tag":"","thread_warp_size":32},
        [["TENSOR","int8",[1024,1024]],["TENSOR","int8",[1024,1024]],["TENSOR","int32",[1024,1024]]]]]
    ```
  - schedule_2
    ```
        [0,[[[
        ["GetBlock",[],["compute","main"],["b0"]],
        ["GetBlock",[],["root","main"],["b1"]],
        ["Annotate",["b0","\"SSSRRSRS\""],["meta_schedule.tiling_structure"],[]],
        ["GetLoops",["b0"],[],["l2","l3","l4"]],
        ["Split",["l4","None",4],[1],["l5","l6"]],
        ["Reorder",["l6"],[],[]],
        ["Blockize",["l6"],[true],["b7"]],
        ["Annotate",["b7","\"dp4a\""],["meta_schedule.auto_tensorize"],[]],
        ["GetLoops",["b7"],[],["l8","l9","l10"]],
        ["SamplePerfectTile",["l8"],[5,64],["v11","v12","v13","v14","v15"]],
        ["Split",["l8","v11","v12","v13","v14","v15"],[1],["l16","l17","l18","l19","l20"]],
        ["SamplePerfectTile",["l9"],[5,64],["v21","v22","v23","v24","v25"]],
        ["Split",["l9","v21","v22","v23","v24","v25"],[1],["l26","l27","l28","l29","l30"]],
        ["SamplePerfectTile",["l10"],[3,64],["v31","v32","v33"]],
        ["Split",["l10","v31","v32","v33"],[1],["l34","l35","l36"]],
        ["Reorder",["l16","l26","l17","l27","l18","l28","l34","l35","l19","l29","l36","l20","l30"],[],[]],
        ["Fuse",["l16","l26"],[1],["l37"]],
        ["Bind",["l37"],["blockIdx.x"],[]],
        ["Fuse",["l17","l27"],[1],["l38"]],
        ["Bind",["l38"],["vthread.x"],[]],
        ["Fuse",["l18","l28"],[1],["l39"]],
        ["Bind",["l39"],["threadIdx.x"],[]],
        ["Annotate",["b7",32],["meta_schedule.thread_extent_low_inclusive"],[]],
        ["Annotate",["b7",1024],["meta_schedule.thread_extent_high_inclusive"],[]],
        ["CacheWrite",["b7",[]],[0,"local"],["b40"]],
        ["ReverseComputeAt",["b40","l39"],[1,-1],[]],
        ["CacheRead",["b7",["b7"]],[0,"shared"],["b41"]],
        ["ComputeAt",["b41","l34"],[1,-1],[]],
        ["GetLoops",["b41"],[],["l42","l43","l44","l45","l46","l47"]],
        ["Fuse",["l46","l47"],[1],["l48"]],
        ["SampleCategorical",[],[[1,2,3,4],[0.25,0.25,0.25,0.25]],["v49"]],
        ["Annotate",["b41","v49"],["meta_schedule.cooperative_fetch"],[]],
        ["CacheRead",["b7",["b7"]],[1,"shared"],["b50"]],
        ["ComputeAt",["b50","l34"],[1,-1],[]],
        ["GetLoops",["b50"],[],["l51","l52","l53","l54","l55","l56"]],
        ["Fuse",["l55","l56"],[1],["l57"]],
        ["SampleCategorical",[],[[1,2,3,4],[0.25,0.25,0.25,0.25]],["v58"]],
        ["Annotate",["b50","v58"],["meta_schedule.cooperative_fetch"],[]],
        ["SampleCategorical",[],[[0,16,64,512,1024],[0.2000000000000000111,0.2000000000000000111,0.2000000000000000111,0.2000000000000000111,0.2000000000000000111]],["v59"]],
        ["Annotate",["b1","v59"],["meta_schedule.unroll_explicit"],[]],
        ["EnterPostproc",[],[],[]],
        ["Unannotate",["b41"],["meta_schedule.cooperative_fetch"],[]],
        ["GetLoops",["b41"],[],["l60","l61","l62","l63","l64"]],
        ["Split",["l64","None",1024,4],[1],["l65","l66","l67"]],
        ["Vectorize",["l67"],[],[]],["Bind",["l66"],["threadIdx.x"],[]],
        ["Unannotate",["b50"],["meta_schedule.cooperative_fetch"],[]],
        ["GetLoops",["b50"],[],["l68","l69","l70","l71","l72"]],
        ["Split",["l72","None",1024,2],[1],["l73","l74","l75"]],
        ["Vectorize",["l75"],[],[]],
        ["Bind",["l74"],["threadIdx.x"],[]],
        ["GetBlock",[],["root","main"],["b76"]],
        ["Unannotate",["b76"],["meta_schedule.unroll_explicit"],[]],
        ["GetChildBlocks",["b76"],[],["b77","b78","b79","b80"]],
        ["GetLoops",["b77"],[],["l81","l82","l83","l84","l85","l86","l87"]],
        ["GetLoops",["b78"],[],["l88","l89","l90","l91","l92","l93","l94"]],
        ["GetLoops",["b79"],[],["l95","l96","l97","l98","l99","l100","l101","l102","l103","l104"]],
        ["GetLoops",["b80"],[],["l105","l106","l107","l108","l109"]],
        ["GetBlock",[],["compute_o","main"],["b110"]],
        ["GetLoops",["b110"],[],["l111","l112","l113","l114","l115","l116","l117","l118","l119","l120"]],
        ["DecomposeReduction",["b110","l114"],[],["b121"]],
        ["Unannotate",["b121"],["meta_schedule.auto_tensorize"],[]],
        ["Annotate",["b121","\"\""],["meta_schedule.auto_tensorize"],[]],
        ["GetBlock",[],["compute_o_update","main"],["b122"]],
        ["Unannotate",["b122"],["meta_schedule.auto_tensorize"],[]],
        ["Tensorize",["b122"],["dp4a",true],[]]],
        [[9,[2,2,32,1,8]],[11,[1,2,32,16,1]],[13,[64,4,1]],[30,3],[36,1],[38,0]]],
        [10000000000],
        {"arch":"sm_86","keys":["cuda","gpu"],"kind":"cuda","max_num_threads":1024,"max_shared_memory_per_block":49152,"max_threads_per_block":1024,"registers_per_block":65536,"tag":"","thread_warp_size":32},
        [["TENSOR","int8",[1024,1024]],["TENSOR","int8",[1024,1024]],["TENSOR","int32",[1024,1024]]]]]
    ```

- database_workload.json
  - - workload table
  - ./python/tvm/meta_schedule/database/json_database.py
- logs
  - tvm.meta_schedule.logging.task_0_fused_nn_dense.log
    - task_scheduler.cc:159/TaskSchedulerNode::Tune
      - Initializing Task #0
      - Design space #0,1,2
    - evolutionary_search.cc
      - Generating candidates
        - XGB
          - XGB iter  50: tr-p-rmse: 0.000000	tr-a-peak@32: 1.000000	tr-rmse: 0.750000	tr-rmse: 0.750000
          - XGB stopped. Best iteration: [9] tr-p-rmse:0.00000	tr-a-peak@32:1.00000	tr-rmse:0.75000	tr-rmse:0.75000 n
        - Evolve iter #0,1,2,3
      - Sending 32 candidates(s) for measurement
    - task_scheduler.cc:193/TaskSchedulerNode::Tune
      - [Task #0: fused_nn_dense] Trial #1
        - @I.ir_module
        - Error in building
          - LocalBuilder: Timeout, killed after 30.0 seconds
          - error: identifier "__dp4a" is undefined
        - Error in running
          - CUDA_ERROR_NO_BINARY_FOR_GPU
  - tvm.meta_schedule.logging.task_scheduler.log
    - task_scheduler.cc:320/TaskSchedulerNode::PrintTuningStatistics
## llvm.ir
- auto_tensorize.ll
