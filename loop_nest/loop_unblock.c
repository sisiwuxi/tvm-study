#include<stdio.h>
#include<stdlib.h>
#include<x86intrin.h>
#include<time.h>

#define N (512)// (512) (256)
#define M (1024)
#define SI (8)
#define MIN(x,y) (((x)<(y))?(x):(y))

int loop_unblock_1() {
  int i,j;
  float A[N][M], B[N];
  clock_t start,finish;
  double total_time;

  for (i=0; i<N; i++) {
    B[i] = rand()%10;
    for (j=0; j<M; j++) {
      A[i][j] = rand()%10;
    }
  }
  start = clock();
  for (j=0; j<M; j++) {
    for (i=0; i<N; i++) {  
      B[i] = B[i] + A[i][j];
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_unblock_1 total_time: %f seconds\n", total_time);
  return 0;
}


int loop_unblock_2() {
  int i,j,ii,ji;
  float A[N][M], B[N];
  clock_t start,finish;
  double total_time;

  for (i=0; i<N; i++) {
    B[i] = rand()%10;
    for (j=0; j<M; j++) {
      A[i][j] = rand()%10;
    }
  }

  start = clock();
  for (i=0; i<N; i+=SI) {
    for (j=0; j<M; j++) {  
      for (ii=i; ii<MIN(i+SI, N); ii++) {
        B[ii] = B[ii] + A[ii][j];
      }
    }
  }
  finish = clock();

  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_unblock_2 total_time: %f seconds\n", total_time);
  return 0;
}

/*
loop_unblock_1 total_time: 0.001636 seconds
loop_unblock_2 total_time: 0.001202 seconds
*/
void test_vec_add_matrix() {
  loop_unblock_1();
  loop_unblock_2();
  return;
}


int loop_unblock_3() {
  int i,j,k;
  double A[N][N], B[N][N], C[N][N];
  clock_t start,finish;
  double total_time;

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      A[i][j] = rand()%10;
      B[i][j] = rand()%10;
      C[i][j] = rand()%10;
    }
  }

  start = clock();
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      for (k=0; k<N; k++) {
        C[i][j] = C[i][j] + A[i][k]*B[k][j];
      }
    }
  }
  finish = clock();

  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_unblock_3 total_time: %f seconds\n", total_time);
  return 0;
}

int loop_unblock_4() {
  int i,j,k,ii,ji,ki;
  double A[N][N], B[N][N], C[N][N];
  clock_t start,finish;
  double total_time;

  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      A[i][j] = rand()%10;
      B[i][j] = rand()%10;
      C[i][j] = rand()%10;
    }
  }

  int S = 4; // 4, 8, 16
  start = clock();
  for (i=0; i<N; i+=S) {
    for (j=0; j<N; j+=S) {
      for (k=0; k<N; k+=S) {
        for (ii=i; ii<MIN(i+S, N); ii++) {
          for (ji=j; ji<MIN(j+S, N); ji++) {
            for (ki=k; ki<MIN(k+S, N); ki++) {
              C[ii][ji] = C[ii][ji] + A[ii][ki]*B[ki][ji];
            }
          }
        }
      }
    }
  }
  finish = clock();

  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_unblock_4 total_time: %f seconds\n", total_time);
  return 0;
}

/*
loop_unblock_3 total_time: 0.056960 seconds
loop_unblock_4 total_time: 0.074162 seconds
*/
void test_dot() {
  loop_unblock_3();
  loop_unblock_4();
  return;
}

/*
clang loop_unblock.c
*/
int main() {
  // test_vec_add_matrix();
  test_dot();
  return 0;
}