#include<stdio.h>
#include<stdlib.h>
#include<x86intrin.h>
#include<time.h>

// #define M (4)
// #define N (512)
#define M (16)
#define N (64)
#define K (64)
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

int loop2_lean_before() {
  float A[N][N];
  clock_t start,finish;
  double total_time;

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = rand()%10;
    }
  }
  start = clock();
  for (int i=1; i<N; i++) {
    for (int j=1; j<N; j++) {
      A[i][j] = A[i][j-1] + A[i-1][j];
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop2_lean_before total_time: %f seconds\n", total_time);
  return 0;
}

int loop2_lean_after() {
  float A[N][N];
  clock_t start,finish;
  double total_time;

  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = rand()%10;
    }
  }
  start = clock();
  for (int j=2; j<2*N; j++) {
    for (int i=MAX(1,j-N+1); i<MIN(N,j); i++) {
      A[i][j-i] = A[i-1][j-i] + A[i][j-i-1];
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop2_lean_after total_time: %f seconds\n", total_time);
  return 0;
}

void test_loop2_lean() {
  loop2_lean_before();
  loop2_lean_after();
  return;
}


int loop3_lean_before() {
  float A[M][N][K], B[M][N][K];
  clock_t start,finish;
  double total_time;

  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<K; k++) {
        A[i][j][k] = 1;
        B[i][j][k] = 2;
      }
    }
  }
  start = clock();
  for (int i=2; i<M+1; i++) {
    for (int j=2; j<N+1; j++) {
      for (int k=1; k<K; k++) {
        A[i][j][k] = A[i][j-1][k] + A[i-1][j][k];
        B[i][j][k] = B[i][j][k] + A[i][j][k];
      }
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop2_lean_before total_time: %f seconds\n", total_time);
  return 0;
}

int loop3_lean_after() {
  float A[M][N][K], B[M][N][K];
  clock_t start,finish;
  double total_time;

  for (int i=0; i<M; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<K; k++) {
        A[i][j][k] = 1;
        B[i][j][k] = 2;
      }
    }
  }
  start = clock();
  for (int k=2; k<N+K; k++) {
    for (int i=MAX(1,k-N-K); i<MIN(M,k+K-2); i++) {
      for (int j=MAX(1,k-i-K+1); j<MIN(N,k+i-1); j++) {
        A[i][j][k-i-j] = A[i][j-1][k-i-j] + A[i-1][j][k-i-j];
        B[i][j][k-i-j] = B[i][j][k-i-j] + A[i][j][k-i-j];
      }
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop2_lean_before total_time: %f seconds\n", total_time);
  return 0;
}

void test_loop3_lean() {
  loop3_lean_before();
  loop3_lean_after();
  return;
}

/*
$ gcc loop_lean.cc -O3 -fopt-info
$ icc before.cc -O3 -qopt-report=5; loop_lean.optrpt
$ icc after.cc -O3 -vec-threshold0 -qopt-report=5
  - vectorization support
*/
int main() {
  // test_loop2_lean();
  test_loop3_lean();
  return 0;
}