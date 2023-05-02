# test
- meta_schedule.auto_tensorize
- tvm/test/python/integration
  - test_auto_tensorize.py
  - test_tuning.py
- tvm/tests/python/unittest
  - test_meta_schedule_*
  - test_meta_schedule_byoc_tensorrt.py
  - test_meta_schedule_cpu_dot_product.py
  - test_meta_schedule_postproc_rewrite_tensorize.py
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
# database_workload.json
# logs/tvm.meta_schedule.logging.task_0_main.log
  - task_scheduler.cc
  - evolutionary_search.cc
# logs/tvm.meta_schedule.logging.task_scheduler.log
  - class Module

# dataset_collect_models
- python3 dataset_collect_models.py --model_cache_dir="./del"

# dataset_sample_candidates
- tvm/python/tvm/meta_schedule/testing/dataset_sample_candidates.py
- python3 dataset_sample_candidates.py --task_cache_dir="./del" --candidate_cache_dir="./del"
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

