#include<stdio.h>
#include<time.h>

int llvm_licm() {
  // clang loop_invariance.c -O1 -Rpass=licm
  const int M = 256;
  const int N = 256;
  float U[M], W[M], D[M];
  float dt = 5.0;
  clock_t start,finish;
  double total_time;
  for (int i=1; i<N; i++) {
    U[i] = i;
    W[i] = i+1;
    D[i] = i+2;
  }

  start = clock();
  for (int i=1; i<N; i++)
    for (int j=1; j<M; j++)
      U[i] = U[i] + W[i]*W[i]*D[j]/(dt*dt);
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("dot_before total_time: %f seconds\n", total_time);
  // loop_invariance.c:17:21: remark: hoisting load [-Rpass=licm]
  // loop_invariance.c:17:25: remark: hoisting fmul [-Rpass=licm]
  //       U[i] = U[i] + W[i]*W[i]*D[j]/(dt*dt);
  //                         ^
  // loop_invariance.c:17:12: remark: Moving accesses to memory location out of the loop [-Rpass=licm]
  //       U[i] = U[i] + W[i]*W[i]*D[j]/(dt*dt);

  start = clock();
  float T1 = 1/(dt*dt);
  for (int i=1; i<N; i++) {
    float T2 = W[i]*W[i];
    for (int j=1; j<M; j++)
      U[i] = U[i] + T2*D[j]*T1;
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("dot_before total_time: %f seconds\n", total_time);

  printf("%f", U[1]);
}


int llvm_cmp() {
  // - clang -emit-llvm -S loop_invariance.c -fno-discard-value-names
  // - clang -O1 -emit-llvm -S loop_invariance.c -fno-discard-value-names -o 1-opt.ll
  const int M = 256;
  const int N = 256;
  float U[M], W[M], D[M];
  float dt = 5.0;
  for (int i=1; i<N; i++) {
    U[i] = i;
    W[i] = i+1;
    D[i] = i+2;
  }

  for (int i=1; i<N; i++)
    for (int j=1; j<M; j++)
      U[i] = U[i] + W[i]*W[i]*D[j]/(dt*dt);
}

int main() {
  // llvm_licm();
  llvm_cmp();
}