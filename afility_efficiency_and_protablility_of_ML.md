# afility efficiency and protablility of ML deployment with apache tvm and octoML

# rapidly exploding AI/ML infrastructure ecosystem
- rapidly evolving models and ML framework landscape
  - 85000 models
  - cambrian explosion of HW backends

# problems
## a pretty windy long messy path betwwen the models and the hardware
- largely interpreted underpinned by hand-written libraries
- use-case or HW-specific stack often hand-written
- painful and unproductive for users
- brittle for application writers
- unsustainable for HW/platform vendors: need to keep up with model and framework evolution
- models -> accessible -> infrastructure dependence -> operational costs -> applications

## why is hard
- need to unify high-level model optiomizations with low-level HW-dependent code generation/optimizations(fusion, data type and layout)
- new HW features(tensor ops) are difficult to be used automatically
  - google TPU
  - NV tensor core
  - AMD matrix core
  - Intel matrix engine
  - Apple neural engine
  - Arm ethos-N
  - T-head hanguang
- memory hierarchies ever more complex
- the optimization space is gigantic
  - need search-based optimizations from ground up
- fast changing model needs and HW features
  - need to be extensible across the stack

# TVM
- tensor virtual machine
- automated, open source, unified optimization and compilation framework for deep learning
- model in, HW-sprcific, native code out
- interpreted -> compiled
- hand-written kernels -> ML based auto-tuning
- framework & HW dependent -> framework & HW agnostic

# relative work
- mlir dialects
- openXLA
- glow

# a possibly controversial historical analogy
- unlocked fast growth in the HW & SW industries
  - pre-80s: IBM sold tightly coupled HW+SW
  - 80s: mircosoft decoupled the SW from the box
  - 90s: linux open sourced OSs
- unlocked fast growth and innovation in the AI/ML industry
  - pre-2018: ML with proprietary HW+SW
  - 2018: TVM compilers/runtimes decouple ML from the HW

# tvm stack overview
- framework
- relay/relax
- tensor ir
- pre-optimized op libs(cuDNN, MKL-DNN, NNPack, ROCm)
- codegen backends(x86, CUDA, AMD CPU, APU&GPU, ARM CPU, MIPS, RISC-V, FPGA)
- goal: enable access to high-performance machine learning anywhere for everyone. built from the ground up fro automation, with ML and constraint-based search for optimized code.
- enable short time-to-deployment for new models and new HW. easy to extend to new HW, and new models.

# who should use tvm
- ml end user: wants to quick optimize their model on best/chosen HW target
  - ML engineer for production apps
  - product R&D
  - ML researchers
- hw chip vendor & platform provider: wants to offer their customers the best ML SW stack for all models
  - system SW engineers
  - technical sales demos

# a compiler and runtime framework
- end to end DL model optimization
- enabling existing kernel operator libraries to support frameworks
- support rapid hw/sw codesign
  - VTA: bring your own RTL
    - enables rapis feedback loops for SW/HW codesign
    - includeing end-to-end automated SW/HW loops
    - verilator and other EDA simulators supported
  - write custome kernel
    - full control
    - autotuning
    - autoscheduling
    - script for high-level functional spec
  
# standalone bare-metal model
- tinyML

# tvm is an industry standard open source ml stack
- 1B+ devices
- 770 contributors
- 1600+ registrations
- 13% HW vendors
- 35% research
- 50% ML end user

# tvm technology advantages
- ml-driven auto-tuning/codegen
- hw-aware data type: quantization/specialization
- split-graph execution for heterogeneous HW
- bare-metal compilation for self-contained deployment in IoT and browsers(WASM/WebGPU)
- support for sparse tensors(NLP, classical ML, LLMs)
- mix and match with operator libs and native compilers stacks
- integration w/rest of eco-system(mlir dialects)

# ml-based optimizations
- initial program
  - low-level abstract syntax tree(AST)
  - AST is a common representation
- search space
  - statistical features of AST
  - extract hierarchical optimization serach space from naive implementation
- automation
  - automatically adapt to hardware type by learning, transferable to other tasks
- statistical cost model
- code generator
- training data
  - start from repository of existing cost models, augment with more experiments
  - the more it is used, the better it gets

# halide
- functional definition, expression
- how to generate code, schdule

## autotvm
- template based
  - pros
    - allow users to inject domain knowledge to deal with special cases
    - much more efficient than trying every option manually
  - cons
    - templates erquire expertise to write and may have gaps
    - difficult to scale across all operators and target platforms

## autoschduler
- functional definition + schdule rules
  - pros
    - no engineering effort required
    - rules can cover a larger space than users can think of, leading to better results
  - cons
    - difficult to extend rule system
    - does not work well for "special" cased like use of tensorcores

# meta-schdule
- functional definition + functional description of HW
- operates directly on tensorIR to enable both automatic and manual schedule generation
- provided with functional descrption of the HW-native tensor ops, can use them automaticaly
  - kernel
- pros
  - best of all schedule generation approaches
  - tighter integration with tensorIR enables new capabilities like accelerators
  - better debuggability through TVMScript
  - designed for production, many "quality of life" improvements for features like reuseable tuning logs
  - 1.5x~2x compared to hand-tuned and hyper-specialized compilers

# vertical boundaries
- modular, multi-stage compilation is useful bu not sufficient
- modular components
  - compuitational graphs
  - tensor programs
  - hardware primitives

# unify ML optimizations across
- compuitational graphs: relax
- tensor programs: tensorIR
- libriries: FFI
- hardware primitives: auto tensorization
- single abstraction for all key persons increases velocity

## SaaS architecture
- runtimes and libraries
  - metaschedule
- optimization passes
  - data precision
  - data layout
  - fusion and tensor layout
  - thread-pinning and runtime concurrency
  - batch size exploration
- hardware coverage
  - multi-cloud
  - multi-ISA
  - containerizationwith runtime dependencies