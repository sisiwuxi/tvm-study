# outline
- loop unroll
- loop fusion
- loop distribution

# reference
- optimizing compilers for modern architectures a dependence-based approach
- Kai Nacke. Learn LLVM 12, A beginner's guide to learning llvm compiler tools and core libraries with c++


# why
- The most time-consuming program is generally presented as a loop form

---

# Loop unrolling
- Reduce the number of branch instruction
- Increase the space for processor instruction scheduling
- Get more opportunities for instruction parallelism
- Increase the rate of register reuse

## Loop compression
- before:
```c
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = A[i][j] + B[i][j]*C[i][j]
    }
  }
```
- after
```c
  for(i=0; i<N; i++){
    for(j=0; j<N; j+=4){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j]
      A[i][j+1] = A[i][j+1] + B[i][j+1]*C[i][j+1]
      A[i][j+2] = A[i][j+2] + B[i][j+2]*C[i][j+2]
      A[i][j+3] = A[i][j+3] + B[i][j+3]*C[i][j+3]
    }
  }
```

## Illegality
- before:
```c
  for(i=1; i<N; i++){
    for(j=1; j<N; j++){
      A[i+1][j-1] = A[i][j] + 3
    }
  }
```
  - dependent
      j=1             j=2             j=3             j=4
  i=1 A[2][0]=A[1][1] A[2][1]=A[1][2] A[2][2]=A[1][3] A[2][3]=A[1][4]
  i=2 A[3][0]=A[2][1] A[3][1]=A[2][2] A[3][2]=A[2][3] A[3][3]=A[2][4]
  i=3 A[4][0]=A[3][1] A[4][1]=A[3][2] A[4][2]=A[3][3] A[4][3]=A[3][4]
  i=4 A[5][0]=A[4][1] A[5][1]=A[4][2] A[5][2]=A[4][3] A[5][3]=A[4][4]

- after
```c
  for(i=1; i<N; i+=2){
    for(j=1; j<N; j++){
      A[i+1][j-1] = A[i][j] + 3
      A[i+2][j-1] = A[i+1][j] + 3
    }
  }
```
  - dependent
      j=1             j=2             j=3             j=4
  i=1 A[2][0]=A[1][1] A[2][1]=A[1][2] A[2][2]=A[1][3] A[2][3]=A[1][4]
      A[3][0]=A[2][1] A[3][1]=A[2][2] A[3][2]=A[2][3] A[3][3]=A[2][4]
  i=3 A[4][0]=A[3][1] A[4][1]=A[3][2] A[4][2]=A[3][3] A[4][3]=A[3][4]
      A[5][0]=A[4][1] A[5][1]=A[4][2] A[5][2]=A[4][3] A[5][3]=A[4][4]
  
  - problem:
  ```c
    A[3][0]=A[2][1]
      before: A[2][1]=A[1][2], ...
      after:  A[2][1] not update
  ```

## Fully loop unroll
- before
```c
  void loop_unroll(void) {
    float a[8];
    for(int i=0; i<8; i++) {
      a[i] = a[i] + 3;
    }
  }
```
- after
```c
  void loop_unroll(void) {
    float a[8];
    for(int i=0; i<8; i+=8) {
      a[i] = a[i] + 3;
      a[i+1] = a[i+1] + 3;
      a[i+2] = a[i+2] + 3;
      a[i+3] = a[i+3] + 3;
      a[i+4] = a[i+4] + 3;
      a[i+5] = a[i+5] + 3;
      a[i+6] = a[i+6] + 3;
      a[i+7] = a[i+7] + 3;
    }
  }
```
- Unroll is not the more the better
- Need more register, Increase the risk of instruction buffer overflow

## Not evenly divisible
- before
```c
  for(i=1; i<512; i++){
    for(j=1; j<510; j+=3){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j]
      A[i][j+1] = A[i][j+1] + B[i][j+1]*C[i][j+1]
      A[i][j+2] = A[i][j+2] + B[i][j+2]*C[i][j+2]
    }
    for(j=510; j<512; j++){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j]
    }
  }
```
- after
```c
  for(i=1; i<512; i++){
    for(j=1; j<504; j+=9){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j]
      A[i][j+1] = A[i][j+1] + B[i][j+1]*C[i][j+1]
      A[i][j+2] = A[i][j+2] + B[i][j+2]*C[i][j+2]
      A[i][j+3] = A[i][j+3] + B[i][j+3]*C[i][j+3]
      A[i][j+4] = A[i][j+4] + B[i][j+4]*C[i][j+4]
      A[i][j+5] = A[i][j+5] + B[i][j+5]*C[i][j+5]
      A[i][j+6] = A[i][j+6] + B[i][j+6]*C[i][j+6]
      A[i][j+7] = A[i][j+7] + B[i][j+7]*C[i][j+7]
      A[i][j+8] = A[i][j+8] + B[i][j+8]*C[i][j+8]            
    }
    for(j=504; j<512; j++){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j]
    }
  }
```
## llvm unroll
- limitation: complex dependencies
  -funroll-loops
  -fno-unroll-loops
  -mllvm -unroll-max-count
  -mllvm -unroll-count
  -mllvm -unroll-runtime
  -mllvm -unroll-threshold
  -mllvm -unroll-remainder
  -Rpass=loop-unroll
  -Rpass-missed=loop-unroll

## progma unroll
  #pragma clang loop unroll(enable)
  #pragma clang loop unroll(full)
  #pragma clang loop unroll_count(8)


---


# loop fusion:
- The process of merging two loops with the same iteration space into one loop 
- Belongs to the loop transformation at the statement level

## example_1
- before
  for(i = 0; i<N; i++)
    x[i] = a[i] + b[i];
  for(i = 0; i<N; i++)
    y[i] = a[i] - b[i];
- after
  for(i = 0; i<N; i++)
    x[i] = a[i] + b[i];
    y[i] = a[i] - b[i];

## pros
- Reduce the cost of the loop iteration
- Enhanced the data(a, b) reuse and register reuse
- Reduce the cost of startup parallel optimization and eliminate the cost of thread synchronization between multiple loops before loop fusion
- Increased the scope for loop optimization

## legality
  - Cannot violate the original dependency, if there is a loop-independent dependency path between two loops, this path contains a loop statement that is not merged with them, then they cannot be fusion
  - No new dependencies can be generated, and if there is a merge-blocking dependency between two loops, they cannot be fusion

## example_2: illegality
  - before
    for(i = 1; i<N; i++)
      a[i] = b[i] + c; //s1
    for(i = 1; i<N; i++)
      d[i] = a[i+1] + e; //s2
    - s1 -> s2: loop-independent dependency

  - after
    for(i = 1; i<N; i++)
      a[i] = b[i] + c; //s1
      d[i] = a[i+1] + e; //s2
    - s2 -> s1: loop carries anti-loop

## benefits of loop fusion
- An important application scenario of loop fusion is parallelism, but not all loop fusion can bring benefits to parallelism

## seperation limitation
  - when a parallel loop and a serial loop are fusion, the lopp must be executed serially after fusion
  - before
    for(i = 1; i<N; i++)
      a[i] = b[i] + 1; //s1
    for(i = 1; i<N; i++)
      c[i] = a[i] + c[i-1]; //s2
    - s1: parallism
    - s2: serial
  - after
    for(i = 1; i<N; i++)
      a[i] = b[i] + 1; //s1
      c[i] = a[i] + c[i-1]; //s2
    - s1 & s2: serail -> slow down

## Dependency limitation that prevent parallelism
  - When two loops that can be parallelized have a dependency that prevents parallelism, the dependency is carried by the merged loop after loop fusion
  - before
    for(i = 1; i<N; i++)
      a[i+1] = b[i] + c; //s1
    for(i = 1; i<N; i++)
      d[i] = a[i] + e; //s2
  - s1: parallel
  - s2: parallel
  - s1 -> s2: loop carry dependency
  - after
    for(i = 1; i<N; i++)
      a[i+1] = b[i] + c; //s1
      d[i] = a[i] + e; //s2

## loop fusion in compiler
### Loop fusion needs to meet the following conditions in llvm 
- The two loops must be adjacent, and there cannot be statements between the two loops
- Both loops must have the same number of iterations
- Loops must be equivalent control flow, if one loop executes, the other loop also executes
  - example_dismatch
    ```
           loop1
             |
            \|/
            block2
            /\
           /  \
        block3  \
          |     block4
         \|/       /
        loop4     /
          |      /
         \|/    /
        block5 /
          |   /
         \|/ /
        block7
    ```
    - loop4 in a branch statement, loop4 is executable or not
  - example_match
    ```
           loop1
             |
            \|/
           block2
            /  \
          \//  \\/
       block3  block4
          \      /
          \\/  \//
           block5
             |
            \|/
           loop6
    ```
    - The loop branch statement starts with loop1 and ends with loop6
    - It can be guaranteed that when loop1 is executed, loop6 will be executed
- There cannot be negative distance dependencies between loops. For example, when the number of iterations of the second loop is m, use the value calculated by the first loop in the future m+n iterations
  - example_dismatch
  for(i=1; i<N; i++)
    a[i] = b[i] + c; //Lj
  for(i=1; i<N; i++)
    d[i] = a[i+4] + e //Lk
    - i=1, Lk need a[5]
- options
  - -fproactive-loop-fusion-analysis: Turn on loop fusion & analysis
  - -fproactive-loop-fusion: Turn on loop fusion
  - -loop-fusion: optimize the intermediate representation through the opt tool
    - cp llvm-project/llvm/test/Transforms/LoopFusion/simple.ll loop_fusion.ll
    - opt -S -loop-fusion loop_fusion.ll &> loop_fusion_opt.ll
    

---

# loop distribute
- Distribute one loop into multiple loops, and each loop has the same iteration space as the original loop
- the new generated loops only contains a subset of statements from the original loop
- It is often used to distribute vectorizable, tensorizable or parallelizable loops, and then convert the code of the vectorizable part into vector instruction, tensorizable part into tensor instruction

## example_1
- before
```c
  for(int i=0; i<n; i++>) {
    a[i] = i;
    b[i] = 2 + b[i];
    c[i] = 3 + c[i-1];
  }
```
  - c[i] = 3 + c[i-1];
  - loop carrying dependencies
- after
```c
  for(int i=0; i<n; i++>) {
    a[i] = i;
    b[i] = 2 + b[i];
  }
  // barrier
  for(int i=0; i<n; i++>) {
    c[i] = 3 + c[i-1];
  }
```

## pros
- Turn a serial loop into multiple parallel loops
- Implement partial parallelization of loops
- Increased the scope of loop optimization

## cons
- Reduce the granularity of parallelism
- Add additional cost of communication and synchronization

## example_2
- before
```c
  for(int i=1; i<N; i++>) {
    for(int j=1; j<N; j++>) {
      A[i][j] = B[i][j] + C[i][j]; //s1
      D[i][j] = A[i][j-1]*2; //s2
    }
  }
```
  - s1 and s2 can vectorize
  - but s2 depend the resule of s1
- after
```c
  for(int i=1; i<N; i++>) {
    for(int j=1; j<N; j++>) {
      A[i][j] = B[i][j] + C[i][j]; //s1
    }
  }
  for(int i=1; i<N; i++>) {
    for(int j=1; j<N; j++>) {
      D[i][j] = A[i][j-1]*2; //s2
    }
  }  
```
  - s1 and s2 can parallel

## perfect loop nest
- If the loop is not a tightly nested loop, the subsequent optimization operations cannot be performed, the loop distribution can be used to transform the loop body into a tightly nested loop

## example_dot
- before
```c
  for(int i=1; i<N; i++>) {
    for(int j=1; j<N; j++>) {
      A[i][j] = D; //s1
      for(int k=1; k<N; k++>) {
        A[i][j] = A[i][j] + B[i][k]*C[k][j]; //s2
      }
    }
  }
```
  - premute the layer j and layer k for high access speed and do  vectorize
  - but s1 make the whole

- after
```c
  for(int i=1; i<N; i++>) {
    for(int j=1; j<N; j++>) {
      A[i][j] = D; //s1
    }
  }
  for(int i=1; i<N; i++>) {
    for(int j=1; j<N; j++>) {      
      for(int k=1; k<N; k++>) {
        A[i][j] = A[i][j] + B[i][k]*C[k][j]; //s2
      }
    }
  }
```
  - s1 and s2 are indepent, s2 can do permute j and k now

## example_3
- before
```c
  for(int i=1; i<N; i++>) {
    A[i] = B[i] + 1; //s1
    C[i] = A[i] + C[i-1]; //s2
    D[i] = A[i] + x; //s3
  }
```
- s1: loop distribution
```c
  for(int i=1; i<N; i++>) {
    A[i] = B[i] + 1; //s1
  }
  for(int i=1; i<N; i++>) {
    C[i] = A[i] + C[i-1]; //s2
  }
  for(int i=1; i<N; i++>) {
    D[i] = A[i] + x; //s3
  }
```
- s2: loop fusion
```c
  for(int i=1; i<N; i++>) {
    A[i] = B[i] + 1; //s1
    D[i] = A[i] + x; //s3
  }
  for(int i=1; i<N; i++>) {
    C[i] = A[i] + C[i-1]; //s2
  }
```

## loop_distribution vs loop_fusion
- cache size
- register size

| item | loop_distribution | loop_fusion |
| --- | --- | --- |
| loop judgment conditional execution | Increase | Decrease |
| the number of loop |  1 -> N | N -> 1 |
| loops point to the same array | not good | data reuse |
| one loop use multiply array | increase locality | not good |

## llvm loop distribution
- focus on dependent
- -mllvm -enable-loop-distribute
- gcc: -ftree-loop-distribution

## pragma loop distribution
- # pragma clang loop distribute(enable)

---

# loop permute
- When a loop body contains more than one loop, and no other statements are included between the loop statements, the loop is called a tightly nested loop or perfect loop nest
- Swapping the order of two loops in a tight nest is one of the most effective transformations to improve the program performance
- Loop transformation is a reordering transformation, which only changes the execution order of parameterized iterations
- not delete any statements or generate any new statements, so the legality of loop exchange needs to be judged by the dependencies of the loop

## example accept
- before
```c
  for(int j=0; j<N; j++>) {      
    for(int k=0; k<N; k++>) {
      for(int i=0; i<N; i++>) {
        A[i][j] = A[i][j] + B[i][k]*C[k][j]; //s2
      }
    }
  }
```
  - bad data locality
  - access the next data needs to span a row
- after
```c
  for(int j=0; j<N; j++>) {      
    for(int i=0; i<N; i++>) {    
      for(int k=0; k<N; k++>) {
        A[i][j] = A[i][j] + B[i][k]*C[k][j]; //s2
      }
    }
  }
```
  - read data in B become serial access
  - A[i][j] is constant
  - improve data localityy

## pros
- increase the data locality
  - stride = 1, serial access haas the best data locality
  - the innermost loop decide which index will be accessed by serial order
- increase the recognize of vectorize and parallism
  - access to array B relative to k are serial
  - generate vectorize instruction

## advantage
- example_1
  - before
  ```c
  for(int i=1; i<N; i++>) {
    for(int j=1; j<N; j++>) {
      A[i][j+1] = A[i][j] + 2;
    }
  }
  ```

  - after
  ```c
  for(int j=1; j<N; j++>) {
    for(int i=1; i<N; i++>) {
      A[i][j+1] = A[i][j] + 2;
    }
  }
  ```

- example_2
  - before
  ```c
  for(int j=1; j<N; j++>) {
    for(int i=1; i<M; i++>) {
      A[i][j] = A[i-1][j];
    }
  }
  ```
  - access
  ```
    ------> i
    |
    |   A[1][1] = A[0][1], A[2][1] = A[1][1], A[3][1] = A[2][1], 
    |   A[1][2] = A[0][2], A[2][2] = A[1][2], A[3][2] = A[2][2], 
    |   A[1][3] = A[0][3], A[2][3] = A[1][3], A[3][3] = A[2][3], 
    |
   \|/
  
    j

  ```
  - read and write are not sequential

  - after
  ```c
  for(int i=1; i<M; i++>) {
    for(int j=1; j<N; j++>) {
      A[i][j] = A[i-1][j];
    }
  }
  ```
  - access
  ```
    ------> j
    |
    |   A[1][1] = A[0][1], A[1][2] = A[0][2], A[1][3] = A[0][3], 
    |   A[2][1] = A[1][1], A[2][2] = A[1][2], A[2][3] = A[1][3], 
    |   A[3][1] = A[2][1], A[3][2] = A[2][2], A[3][3] = A[2][3], 
    |
   \|/
  
    i

  ```
  - read and write ares sequential
  - after swap, the innermost not carry dependency, so the statement can do vectorize and parallel

## legality
- It is illegal to do loop permute on an endpoint with dependency
- Do not cause dependency inversion
- before
  ```
  for(j=1; j<N; j++)
    for (i=1; i<N; i++)
      A[i][j+1] = A[i+1][j] + 2;
  ```
  - access
  ```
     ------> i
    |
    |   A[1][2] = A[2][1]+2, A[2][2] = A[3][1]+2, A[3][2] = A[3][1]+2,
    |                        /                    /
    |                      \//                  \//
    |   A[1][3] = A[2][2]+2, A[2][3] = A[3][2]+2, A[3][3] = A[3][2]+2,
    |                       /                    /
    |                     \//                  \// 
    |   A[1][4] = A[2][3]+2, A[2][4] = A[3][3]+2, A[3][4] = A[3][3]+2,
    |
   \|/
  
    j

  ```

- after
  ```
  for (i=1; i<N; i++)
    for(j=1; j<N; j++)
      A[i][j+1] = A[i+1][j] + 2;
  ```
  - access
  ```
     ------> i
    |
    |   A[1][2] = A[2][1]+2, A[2][2] = A[3][1]+2, A[3][2] = A[3][1]+2,
    |                     //\                  //\
    |                     /                    /
    |   A[1][3] = A[2][2]+2, A[2][3] = A[3][2]+2, A[3][3] = A[3][2]+2,
    |                     //\                  //\ 
    |                     /                    /
    |   A[1][4] = A[2][3]+2, A[2][4] = A[3][3]+2, A[3][4] = A[3][3]+2,
    |
   \|/
  
    j

  ```
## llvm
- -mllvm -enable-loopinterchange
- gcc: -floop-interchange
- llvm-project/llvm/test/Transforms/LoopInterchange


---

# loop invariance
- Loop invariants refer to variables whose values do not change in the loop iteration space
- Therefore, it can be mentioned outside of the loop and calculated only once to avoid repeated calculation in the loop

## example
- before
```
  for (int i=1; i<N; i++)
    for (int j=1; j<M; j++)
      U[i] = U[i] + W[i]*W[i]*D[j]/(dt*dt);
```
- after
```
  T1 = 1/(dt*dt)
  for (int i=1; i<N; i++) {
    T2 = W[i]*W[i]
    for (int j=1; j<M; j++)
      U[i] = U[i] + T2*D[j]*T1;
  }
```

## legality
- The transformation cannot affect the semantics of the source program

## pros
- Reduced the calculation strength

## cons
- Not too much computation reduction
- Need occupancy a private register for the variance that moved outside of the loop, so decrease the number of register which the inner loop can used

## llvm
- clang loop_invariance.c -O1 -Rpass=licm
- algorithm
  - Determine cycle
    - calculate the header node, the dominate node and the define node
    - find out the loop nest
    - find out the exit block which have successor out of the loop
  - Match the conditions and moving out of the loop
    - is loop invariance
    - be located in the dominate exit block
    - be located in the basic block in all compute block in the dominated loop 
    - this variane evaluated once
  - DFS
    - adopt the depth first algorithm to search and select it from candidates
    - if all the invariance have been moved outside, then move curent invariance to the preheader block

- loop structure in llvm ir
  - header
    - iteration times, decide to start and stop
  - rest of loop
    - compute in the loop
  - exit
    - other caompute after loop
  - preheader
    - pre executed code
    - The only successor of this block is the header block of the loop

- dominate
  ```
     s1 x1   s2 x2   s3 x3
        \     |     /
         \    |    /
          \   |   /
      s4 Y4=sigma(x1,x2,x3)
              |
             \|/
            s5 Y5
  ```
  - single output node
  - when the program run at Y4, the program in x1 has been run, so x1 is a dominate node
  - example of can not moving out
  ```
    s1: A=B
    s2: B=C+1
  ```
  - A valued one time
  - S2 is not dominate S1, so cannot moving

## example
- before
```
          start
            |-------
           \|/     |
          x=y+1    |  header
           /\      |
          /  \     |
        a=2  a=3   |
          \  /     |
           \/      |
         z=x+1     |
           |--------  
          \|/
          exit
```
- after
```
        x=y+1, z=x+1    preheader
            |
           \|/
          start
            |-------
           \|/     |
           ...     |  header
           /\      |
          /  \     |
        a=2  a=3   |
          \  /     |
           \/      |
          ...      |
           |--------  
          \|/
          exit
```

## example ir
- before
```
  define void @func(i32 %d) {
  Entry:
    br label %Loop
  Loop:
    %j = phi i32 [ 0, %Entry ],[ %Sum, %Loop]
    %loopinvar = add i32 %i, 12
    %Sum = add i32 %j, %loopinvar
    %cond = icmp eq i32 %Sum, 0
    br i1 %cond, label %Exit, label %Loop
  Exit:
    ret void
  }
```
- after
```
  define void @func(i32 %d) {
  Entry:
    %loopinvar = add i32 %i, 12
    br label %Loop
  Loop:
  ;preds = %Loop, %Entry
    %j = phi i32 [ 0, %Entry ],[ %Sum, %Loop]
    %Sum = add i32 %j, %loopinvar
    %cond = icmp eq i32 %Sum, 0
    br i1 %cond, label %Exit, label %Loop
  Exit:
  ;preds = %Loop
    ret void
  }
```
- opt loop_invariance.ll -licm -S -o loop_invariance_opt.ll
```
  ; ModuleID = 'loop_invariance.ll'
  source_filename = "loop_invariance.ll"

  define void @func(i32 %i) {
  Entry:
    %loopinvar = add i32 %i, 12
    br label %Loop

  Loop:                                             ; preds = %Loop, %Entry
    %j = phi i32 [ 0, %Entry ], [ %Sum, %Loop ]
    %Sum = add i32 %j, %loopinvar
    %cond = icmp eq i32 %Sum, 0
    br i1 %cond, label %Exit, label %Loop

  Exit:                                             ; preds = %Loop
    ret void
  }
```

## llvm compare
- clang loop_invariance.c -O1 -Rpass=licm
- clang -emit-llvm -S 1.cpp -fno-discard-value-names
- clang -O1 -emit-llvm -S 1.cpp -fno-discard-value-names -o 1-opt.ll