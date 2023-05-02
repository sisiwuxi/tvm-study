# tvm learning plan
- overall
- schedule primitives


# overall
* user tutorial
https://tvm.apache.org/docs/tutorial/index.html

* schedule primitives - done
https://tvm.apache.org/docs/how_to/work_with_schedules/schedule_primitives.html

* op optimization
https://tvm.apache.org/docs/how_to/optimize_operators/opt_gemm.html#sphx-glr-how-to-optimize-operators-opt-gemm-py

* tensorize
https://tvm.apache.org/docs/how_to/work_with_schedules/tensorize.html#sphx-glr-how-to-work-with-schedules-tensorize-py

* autotvm-op
https://tvm.apache.org/docs/tutorial/autotvm_matmul_x86.html#sphx-glr-tutorial-autotvm-matmul-x86-py

* autoscheduler-op
https://tvm.apache.org/docs/tutorial/auto_scheduler_matmul_x86.html#sphx-glr-tutorial-auto-scheduler-matmul-x86-py

* autotvm-model
https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_x86.html#sphx-glr-how-to-tune-with-autotvm-tune-relay-x86-py

* autoscheduler-model
https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_x86.html#sphx-glr-how-to-tune-with-autoscheduler-tune-network-x86-py

* ir module - skip
https://tvm.apache.org/docs/tutorial/tensor_ir_blitz_course.html#:~:text=An%20IRModule%20is%20the%20central%20data%20structure%20in,an%20IRModule%2C%20which%20can%20be%20created%20from%20TVMScript.

* discuss
https://discuss.tvm.apache.org/

# limitation
- dynamic shape
- TCU
- time costly

# schedule primitives
- tvm/src/te/schedule/schedule_ops.cc
- tvm/src/tir/transforms

# tuning
- algorithm
  - Brute-force
  - Greedy
  - SA, Simulated annealing
  - MCTS, Monte Carlo Tree Search
  - GA, Generic Algorithm
  - Reinforcement Learning
- pretune
  - Lorien
  - https://pypi.org/project/lorien/
- parallel
  - PGRCA(Parallel Genetic RAFT Classification Algorithm)
  - TVM Server Instance（TSI）Scaling
- resouce allocation
  - AdaTune
  - DynaTune
- cost model
  - Gated Neural Network + Normalized Discounted Cumulative Gain
  - A History-Based Auto-Tuning Framework for Fast and High-Performance DNN Design on GPU
  - RAFT(rootfline and fast autotune)
  - FamilySeer: Towards Optimized Tensor Codes by Exploiting Computation Subgraph Similarity
  - TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers
  - MetaTune: Meta-Learning Based Cost Model for Fast and Efficient Auto-tuning Frameworks
  - Simulating Execution Time of Tensor Programs using Graph Neural Networks
  - Tuna: A Static Analysis Approach to Optimizing Deep Neural Networks
  - One-Shot Tuner for Deep Learning Compilers
  - Mind Mappings: Enabling Efficient Algorithm-Accelerator Mapping Space Search
  - ProTuner: Tuning Programs with Monte Carlo Tree Search
  - Bayesian Optimization for auto-tuning GPU kernels

# auto schduler
## ansor
- example
  - tvm/gallery/how_to/tune_with_autoscheduler/tune_conv2d_layer_cuda.py
  - tvm/gallery/tutorial/auto_scheduler_matmul_x86.py
- DietCode: Automatic Optimization for Dynamic Tensor Programs
- Bolt: Bridging the Gap between Auto-tuners and Hardware-native Performance
- transform_step
  - include/tvm/auto_scheduler/transform_step.h
    - AN: AnnotationStep
    - FU: FuseStep
    - PR: PragmaStep
    - RE: ReorderStep
    - SP: SplitStep
    - FSP: FollowSplitStep
    - FFSP: FollowFusedSplitStep
    - SA: StorageAlignStep
    - CA: ComputeAtStep
    - CI: ComputeInlineStep
    - CR: ComputeRootStep
    - CHR: CacheReadStep
    - CHW: CacheWriteStep
    - RF: RfactorStep
  - register customer transform



## meta_schedule
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
  - database_tuning_record.json
    - 32*structure
      - task_id
      - 
  - database_workload.json
  - logs/tvm.meta_schedule.logging.task_0_main.log
    - task_scheduler.cc
    - evolutionary_search.cc
  - logs/tvm.meta_schedule.logging.task_scheduler.log
    - class Module




# Reinforcement Learning
- Optimization of Halide Image Processing Schedules with Reinforcement Learning
- Reinforcement Learning and Adaptive Sampling for Optimized DNN Compilation
- Chameleon: Adaptive Code Optimization for Expedited Deep Neural Network Compilation
- FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System
- Woodpecker-DL: an efficient compiler for accelerating deep learning on heterogeneous computing architectures
- Value Learning for Throughput Optimization of Deep Learning Workloads