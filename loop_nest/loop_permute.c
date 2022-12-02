#include<stdio.h>
#include<time.h>

int locality_1() {
  clock_t start,finish;
  double total_time;
  const int N=512;
  double A[N][N], B[N][N], C[N][N];

  int i,j,k;
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = i;
    }
  }

  start = clock();
  for(int j=0; j<N; j++) {      
    for(int k=0; k<N; k++) {
      for(int i=0; i<N; i++) {
        C[i][j] = C[i][j] + A[i][k]*B[k][j];
      }
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("dot_before total_time: %f seconds\n", total_time);

  start = clock();
  for(int j=0; j<N; j++) {      
    for(int i=0; i<N; i++) {
      for(int k=0; k<N; k++) {    
        C[i][j] = C[i][j] + A[i][k]*B[k][j];
      }
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("dot_after total_time: %f seconds\n", total_time);

  return A[1][2];
}


int locality_2() {
  clock_t start,finish;
  double total_time;
  const int N=512;
  double A[N][N], B[N][N], C[N][N];

  int i,j,k;
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      A[i][j] = j;
    }
  }

  start = clock();
  for(int j=0; j<N; j++) {      
    for(int i=0; i<N; i++) {
      A[i][j] = A[i+1][j];
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("access before total_time: %f seconds\n", total_time);

  start = clock();
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      A[i][j] = A[i+1][j];
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("access after total_time: %f seconds\n", total_time);

  return A[1][2];
}

int loop_interchange() {
  // clang loop_permute.c -O1 -mllvm -enable-loopinterchange -Rpass=loop-interchange -Rpass-missed=loop-interchange -Rpass-analysis=loop-interchange
  const int M=200;
  const int N=300;
  int i,j;
  int A[M][N];
  for (i=0; i<M; i++) {
    for (j=0; j<N; j++) { // remark: Interchanging loops is too costly (cost=1, threshold=0) and it does not improve parallelism. [-Rpass-missed=loop-interchange]
      A[i][j] = j;
    }
  }
  for (j=0; j<N; j++) {
    for (i=0; i<M; i++) { // remark: Loop interchanged with enclosing loop. [-Rpass=loop-interchange]
      A[i][j] = A[i][j]+2;
    }
  }
  return A[1][2];
}

int main() {
  // locality_1();
  // locality_2();
  loop_interchange();
}