# debug
- p sch.mod
- print(sch.mod["main"].script())
- print(tvm.script.asscript(sch.mod["main"]))

# reference
## https://www.youtube.com/watch?v=BKO2tz6D-F8
- trace = instructions + decisions = single new tensor program have different dicisions and different transformations
## https://octoml.ai/blog/introducing-tvms-metaschedule/
- High-level Apache TVM architectural diagram
  - model import
  - optimizations
    - relay passes
      - relay/transformation
    - tuning
      - AutoTVM, AutoScheduler, MetaSchedule
    - tir passes
      - tir/transformation
  - hardware targets
- Apache TVM MetaSchedule core infrastructure diagram
  - tuning trials: generate schedule
  - search space: maximuize relevant breadth
  - cost model: prune efficiently
  - database: maximize reuse
# test
## tests/python/unittest
- test_meta_schedule_*
- test_meta_schedule_byoc_tensorrt.py
- test_meta_schedule_cpu_dot_product.py
- test_meta_schedule_postproc_rewrite_tensorize.py
- test_meta_schedule_arg_info.py
  - TensorInfo
  - str(info)
  - info.as_json()
  - TensorInfo.from_json(info)
  - ArgInfo.from_prim_func(Matmul)
- test_meta_schedule_builder.py
  - test_meta_schedule_single_build
    - artifact_path = tvm_tmp_mod.tar
      - lib0.o: ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), with debug_info, not stripped
      - readelf -a lib0.o
  - test_meta_schedule_multiple_build
    - list[]
  - test_meta_schedule_error_handle_test_builder
    - TestBuilder
    - LocalBuilder
  - test_meta_schedule_error_handle_build_func
    - LocalBuilder(f_export="meta_schedule.builder.test_export", initializer=initializer)
    - LocalBuilder.__init__()
  - test_meta_schedule_error_handle_export_funcs
    - LocalBuilder(f_export="meta_schedule.builder.test_export", initializer=initializer)
  - test_meta_schedule_error_handle_time_out
    - builder = LocalBuilder(timeout_sec=1, f_build="meta_schedule.builder.test_time_out", initializer=initializer,)
- test_meta_schedule_database.py
  - test_meta_schedule_tuning_record_round_trip
    - meta_schedule.TuningRecord
      - trace: Trace
      - run_secs: List[float]
      - workload: Workload
      - target: Target
      - args_info: List[ArgInfo]
    - record.as_json()
      - [[[['GetBlock', [], ['matmul', 'main'], ['b0']], ['GetLoops', ['b0'], [], ['l1', 'l2', 'l3']], ['Split', ['l1', 1, 1, 2, 512], [], ['l4', 'l5', 'l6', 'l7']], ['Split', ['l2', 1, 512, 1, 2], [], ['l8', 'l9', 'l10', 'l11']], ['Split', ['l3', 256, 4], [], ['l12', 'l13']], ['Reorder', ['l4', 'l8', 'l5', 'l9', 'l12', 'l6', 'l10', 'l13', 'l7', 'l11'], [], []]], []], [1.5, 2.5, 1.8], {'kind': 'llvm', 'tag': '', 'keys': ['cpu'], 'link-params': 0}, [['TENSOR', 'float32', [1024, 1024]], ['TENSOR', 'float32', [1024, 1024]], ['TENSOR', 'float32', [1024, 1024]]]]
        - ['Split', ['l1', 1, 1, 2, 512]] -> new loop index = l4, l5, l6, l7
  - test_meta_schedule_database_create
    - JSONDatabase
  - test_meta_schedule_database_add_entry
    - (ret,) = database.get_top_k(workload, 3)
  - test_meta_schedule_database_missing
    - ret = database.get_top_k(workload_2, 3)
  - test_meta_schedule_database_sorting
    - ret = database.get_top_k(token, 2)
  - test_meta_schedule_database_reload
    - new_database = JSONDatabase(path_workload=database.path_workload, path_tuning_record=database.path_tuning_record,)
- test_meta_schedule_integration.py
  - test_meta_schedule_integration_task_extraction_query
    - get_network
    - env.query(task_name="mock-task", mod=mod, dispatched=[MockModule])
  - test_meta_schedule_integration_query_inside_with_scope
    - get_network
    - with env
    - MetaScheduleContext.query_inside_with_scope(task_name="mock-task", mod=mod, dispatched=[MockModule],)
  - test_meta_schedule_integration_extract_from_resnet
    - mod, params, _, _ = get_network
      - modx62 layer
      - paramsx92
        - data, weight, gamma, beta, mean, var, bias
    - extracted_tasks = ms.integration.extract_task(mod, target="llvm", params=params)
- test_meta_schedule_runner.py
  - test_meta_schedule_rpc_single_run
    - RPCRunner(rpc_config, evaluator_config)
  - test_meta_schedule_local_single_run
    - LocalRunner(timeout_sec=100, evaluator_config=evaluator_config)
  - test_meta_schedule_rpc_multiple_runs
    - RPCRunner(rpc_config, evaluator_config)
  - test_meta_schedule_local_multiple_runs
    - LocalRunner(timeout_sec=100, evaluator_config=evaluator_config)
  - test_meta_schedule_py_runner
  - test_meta_schedule_rpc_runner_time_out
    - f_create_session="meta_schedule.runner.test_time_out"
  - test_meta_schedule_local_runner_time_out:
    - @register_func("meta_schedule.runner.test_time_out")
  - test_meta_schedule_rpc_runner_exception
  - test_meta_schedule_local_runner_exception
  - test_meta_schedule_runner_matmul_test
    - RPCRunner(rpc_config, evaluator_config, f_alloc_argument=test_alloc_argument, f_run_evaluator=test_run_evaluator,)
  - test_meta_schedule_runner_add_test
  - test_meta_schedule_local_runner_add_test
- test_meta_schedule_search_strategy.py
  - test_meta_schedule_replay_trace
    - ReplayTrace(num_trials_per_iter=num_trials_per_iter, num_trials_total=num_trials_total)
- test_meta_schedule_space_generator.py
  - test_meta_schedule_space_generator_schedule_fn
  - test_meta_schedule_design_space_generator_union
- test_meta_schedule_task_scheduler.py
  - test_meta_schedule_task_scheduler_single
    - RoundRobin([task], DummyBuilder(), DummyRunner(), database)
    - database
      - python/tvm/meta_schedule/database/database.py:204
  - test_meta_schedule_task_scheduler_multiple
  - test_meta_schedule_task_scheduler_NIE
    - NotImplementedError
  - test_meta_schedule_task_scheduler_override_next_task_id_only
    - MyTaskScheduler(tasks, DummyBuilder(), DummyRunner(), database)
- test_meta_schedule_tune_context.py
  - 
## structure
- python/tvm/tir/schedule/schedule.py
  - type(sch) = class Schedule
    - Utilities
      - mod, state, trace, copy, seed, fork_seed, show
    - Lookup
      - get, get_sref, remove_rv
    - Sampling
      - sample_categorical, sample_perfect_tile
    - Get
      - get_block, get_loops, get_child_blocks, get_producers, get_consumers
    - Transform
      - fuse, split, reorder, parallel, vectorize, bind, unroll, cache_read, cache_write, compute_at, reverse_compute_at, compute_inline, reverse_compute_inline
    - Reduction
      - decompose_reduction, rfactor
    - Block annotation
      - storage_align
    - Blockize & Tensorize
    - Misc
      - enter_postproc
## meta_schedule.auto_tensorize
- tvm/test/python/integration
  - test_auto_tensorize.py
  - test_tuning.py
- tvm/src/meta_schedule
  - task_scheduler/task_scheduler.cc
  - schedule_rule
  - search_strategy
  - postproc
  - mutator
- tvm/python/tvm/tir/tensor_intrin
  - cuda.py
  - x86.py
- test_meta_schedule_tune_tir.py

# application
## transpose
## dense
- alignment
  - m, 16/32
## conv2d
- img2col
- NCHWc
- direct

# log
## database_tuning_record.json
  - max_trials_global * structure
  - task_id
  - features
    - task_scheduler
    - 
    - results
    - target
    - op_info
  """
  [
    0,
    [
      [
        [
          ["GetBlock",[],["update","main"],["b0"]],
          ["GetBlock",[],["root","main"],["b1"]],
          ["Annotate",["b0","\"SSRSRS\""],["meta_schedule.tiling_structure"],[]],
          ["GetLoops",["b0"],[],["l2","l3","l4"]],
          ["SamplePerfectTile",["l2"],[4,64],["v5","v6","v7","v8"]],
          ["Split",["l2","v5","v6","v7","v8"],[1],["l9","l10","l11","l12"]],
          ["SamplePerfectTile",["l3"],[4,64],["v13","v14","v15","v16"]],
          ["Split",["l3","v13","v14","v15","v16"],[1],["l17","l18","l19","l20"]],
          ["SamplePerfectTile",["l4"],[2,64],["v21","v22"]],
          ["Split",["l4","v21","v22"],[1],["l23","l24"]],
          ["Reorder",["l9","l17","l10","l18","l23","l11","l19","l24","l12","l20"],[],[]],
          ["CacheWrite",["b0",[]],[0,"global"],["b25"]],
          ["ReverseComputeAt",["b25","l18"],[1,-1],[]],
          ["Annotate",["b1",64],["meta_schedule.parallel"],[]],
          ["Annotate",["b1",64],["meta_schedule.vectorize"],[]],
          ["SampleCategorical",[],[[0,16,64,512],[0.25,0.25,0.25,0.25]],["v26"]],
          ["Annotate",["b1","v26"],["meta_schedule.unroll_explicit"],[]],
          ["EnterPostproc",[],[],[]],["GetBlock",[],["root","main"],["b27"]],
          ["Unannotate",["b27"],["meta_schedule.parallel"],[]],
          ["Unannotate",["b27"],["meta_schedule.vectorize"],[]],
          ["Unannotate",["b27"],["meta_schedule.unroll_explicit"],[]],
          ["GetChildBlocks",["b27"],[],["b28","b29"]],
          ["GetLoops",["b28"],[],["l30","l31","l32","l33","l34","l35","l36","l37","l38","l39"]],
          ["Fuse",["l30","l31","l32","l33"],[1],["l40"]],
          ["Parallel",["l40"],[],[]],
          ["Fuse",["l39"],[1],["l41"]],
          ["Vectorize",["l41"],[],[]],
          ["Annotate",["l40",64],["pragma_auto_unroll_max_step"],[]],
          ["Annotate",["l40",1],["pragma_unroll_explicit"],[]],
          ["GetLoops",["b29"],[],["l42","l43","l44"]],
          ["Fuse",["l44"],[1],["l45"]],
          ["Vectorize",["l45"],[],[]],
          ["GetBlock",[],["update","main"],["b46"]],
          ["GetLoops",["b46"],[],["l47","l48","l49","l50","l51","l52","l53"]],
          ["DecomposeReduction",["b46","l48"],[],["b54"]]
        ],
        [[4,[2,4,8,2]],[6,[2,2,8,4]],[8,[32,4]],[15,2]]
      ],	
      [0.00012150754394299287549],
      {"keys":["cpu"],"kind":"llvm","num-cores":4,"tag":""},
      [["TENSOR","float32",[128,128]],["TENSOR","float32",[128,128]],["TENSOR","float32",[128,128]]]
    ]
  ]  
  """
## database_workload.json
## logs/tvm.meta_schedule.logging.task_0_main.log
  - task_scheduler.cc
  - evolutionary_search.cc
## logs/tvm.meta_schedule.logging.task_scheduler.log
  - class Module

## dataset_collect_models
- python3 dataset_collect_models.py --model_cache_dir="./del"
  - relay-resnet_18-None-1,3,224,224.json
    - "b64ndarrays": [],
    - "attrs": {"tvm_version": "0.14.dev0"}
    - }<95>^\^@^@^@^@^@^@^@<94>sb<8c>^Hbuiltins....

## dataset_extract_tasks
- python3 dataset_extract_tasks.py --model_cache_dir="./del" --task_cache_dir="./del"
  - resnet_18-
  - relay-resnet_18-None-1,3,224,224_extracted_tasks.json
    - 18*op
      - fused_nn_conv2d_add
      - fused_nn_conv2d_add_1
      - fused_nn_conv2d_add_2
      - fused_nn_conv2d_add_nn_relu
      - fused_nn_max_pool2d
      - fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu
      - fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu
      - fused_nn_conv2d_add_nn_relu_1
      - fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1
      - fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_1
      - fused_nn_conv2d_add_nn_relu_2
      - fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2
      - fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_2
      - fused_nn_conv2d_add_nn_relu_3
      - fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3
      - fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu_3
      - fused_nn_adaptive_avg_pool2d
      - fused_nn_dense_add

## dataset_sample_candidates
- python3 dataset_sample_candidates.py --task_cache_dir="./del" --candidate_cache_dir="./del"
- tvm/python/tvm/meta_schedule/testing/dataset_sample_candidates.py
- main
  - sample_candidates
    - ms.TuneContext
    - context.initialize
    - context.pre_tuning
    - sample_init_population
    - evolve_with_cost_model
    - ms.database.Workload
    - database.commit_workloads
    - database.commit_tuning_record
- AttributeError: <class 'tvm.meta_schedule.tune_context.TuneContext'> has no attribute initialize
