#include<stdio.h>
#include<stdlib.h>
#include<x86intrin.h>
#include<time.h>

#define N (204800)

int loop_strip_1() {
  int i;
  float A[N], B[N], C[N];
  clock_t start,finish;
  double total_time;

  for (i=0; i<N; i++) {
    A[i] = 1;
    B[i] = rand()%10;
    C[i] = rand()%10;
  }
  start = clock();
  for (i=0; i<N; i++) {
    A[i] = B[i] + C[i];
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_strip_1 total_time: %f seconds\n", total_time);
  return A[4];
}


int loop_strip_2() {
  int i, j;
  __m128 vreg0, vreg1, vreg2;
  float A[N], B[N], C[N];
  clock_t start,finish;
  double total_time;

  for (i=0; i<N; i++) {
    A[i] = 1;
    B[i] = rand()%10;
    C[i] = rand()%10;
  }
  int strip = 32;
  start = clock();
  for (i=0; i<N; i+=strip) {
    for (j=i; j<i+strip-1; j+=4) {
      vreg0 = _mm_load_ps(B+j);
      vreg1 = _mm_load_ps(C+j);
      vreg2 = _mm_add_ps(vreg0, vreg1);
      _mm_storeu_ps(A+j, vreg2);
    }
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_strip_2 total_time: %f seconds\n", total_time);
  return A[4];
}

/*
clang loop_strip.c
loop_strip_1 total_time: 0.000957 seconds
loop_strip_2 total_time: 0.000455 seconds
*/
int main() {
  loop_strip_1();
  loop_strip_2();
  return 0;
}