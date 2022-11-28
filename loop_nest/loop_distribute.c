#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define N 1280

int llvm_loop_distribution_before() {
  // clang -O1 -mllvm -enable-loop-distribute loop_distribute.c -Rpass=loop-distribute -Rpass-missed=loop-distribute -Rpass-analysis=loop-distribute -emit-llvm -S
  int A[N], B[N], C[N];
  int i;
  struct timeval time_start, time_end;
  for (i=0; i<N; i++) {
    B[i] = rand();
    C[i] = rand();
  }
  gettimeofday(&time_start, NULL);
  for (i=0; i<N; i++) {
    A[i] = i;
    B[i] = 2 + B[i];
    C[i] = 3 + C[i-1];
  }
  // for (i=0; i<N; i++) {
  //   printf("%d", A[i]);
  //   printf("%d", B[i]);
  //   printf("%d", C[i]);
  // }
  gettimeofday(&time_end, NULL);
  printf("llvm_loop_distribution_before used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
  return 0;
}

int llvm_loop_distribution_after() {
  // clang -O1 -mllvm -enable-loop-distribute loop_distribute.c -Rpass=loop-distribute -Rpass-missed=loop-distribute -Rpass-analysis=loop-distribute -emit-llvm -S
  int A[N], B[N], C[N];
  int i;
  struct timeval time_start, time_end;
  for (i=0; i<N; i++) {
    B[i] = rand();
    C[i] = rand();
  }
  gettimeofday(&time_start, NULL);
  for (i=0; i<N; i++) {
    A[i] = i;
    B[i] = 2 + B[i];
  }
  for (i=0; i<N; i++) {
    C[i] = 3 + C[i-1];
  }
  // for (i=0; i<N; i++) {
  //   printf("%d", A[i]);
  //   printf("%d", B[i]);
  //   printf("%d", C[i]);
  // }
  gettimeofday(&time_end, NULL);
  printf("llvm_loop_distribution_after used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
  return 0;
}

int pragma_loop_distribution() {
  int i;
  // int N = 1024;
  int A[N], B[N], C[N], D[N], E[N];
  struct timeval time_start, time_end;
  for (i=0; i<N; i++) {
    A[i] = i;
    B[i] = i+1;
    D[i] = i+2;
    E[i] = i+3;
  }
  gettimeofday(&time_start, NULL);
  #pragma clang loop distribute(enable)
  for (i=0; i<N; i++) {
    A[i+1] = A[i] + B[i];//s1
    C[i] = D[i]*E[i];//s2
  }
  gettimeofday(&time_end, NULL);
  printf("pragma_loop_distribution used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
  return A[8];
}

int main() {
  llvm_loop_distribution_before();
  llvm_loop_distribution_after();
  pragma_loop_distribution();
  return 0;
}