#include<stdio.h>
#include<stdlib.h>
#include<x86intrin.h>
#include<time.h>

#define N (1024)
#define M (512)

int loop_fission_1() {
  int i;
  int vec[N];
  clock_t start,finish;
  double total_time;

  for (i=0; i<N; i++) {
    vec[i] = i;
  }
  start = clock();
  for (i=0; i<N; i++) {
    vec[i] = vec[i] + vec[M];
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_fission_1 total_time: %f seconds vec[4]=%d\n", total_time, vec[4]);
  return vec[4];
}

int loop_fission_2() {
  int i;
  int vec[N], A[N];
  clock_t start,finish;
  double total_time;

  for (i=0; i<N; i++) {
    vec[i] = i;
  }
  start = clock();
  for (i=0; i<M; i++) {
    vec[i] = vec[i] + vec[M];
  }
  for (i=M; i<N; i++) {
    vec[i] = vec[i] + vec[M];
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_fission_2 total_time: %f seconds vec[4]=%d\n", total_time, vec[4]);
  return vec[4];
}

void test_add() {
  loop_fission_1();
  loop_fission_2();
  return;
}

int loop_fission_3() {
  int i,temp,phi;
  int a[N],b[N],c[N],d[N],coff[N],diff[N];
  clock_t start,finish;
  double total_time;

  temp = 2;
  phi = 2;
  for (i=0; i<N; i++) {
    a[i] = i;
    b[i] = i + 1;
    c[i] = i + 2;
    d[i] = i + 3;
  }
  start = clock();
  for (i=0; i<N; i++) {
    temp = a[i] - b[i];
    coff[i] = (a[i] + b[i]) * temp;
    diff[i+M] = (c[i+M] + d[i+M]) / phi; // error
  }
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_fission_3 total_time: %f seconds\n", total_time);
  // printf("loop_fission_3 total_time: %f seconds diff[M])=%d\n", total_time, diff[M]);
  return 0;
}

int loop_fission_4() {
  int i,temp,phi;
  int a[N],b[N],c[N],d[N],coff[N],diff[N];
  clock_t start,finish;
  double total_time;

  temp = 2;
  phi = 2;
  for (i=0; i<N; i++) {
    a[i] = i;
    b[i] = i + 1;
    c[i] = i + 2;
    d[i] = i + 3;
  }
  start = clock();
  for (i=0; i<N; i++) {
    temp = a[i] - b[i];
    coff[i] = (a[i] + b[i]) * temp;
  }
  for (i=M; i<N; i++) {
    diff[i] = (c[i] + d[i]) / phi;
  }  
  finish = clock();
  total_time = (double)(finish - start)/CLOCKS_PER_SEC;
  printf("loop_fission_4 total_time: %f seconds\n", total_time);
  // printf("loop_fission_4 total_time: %f seconds diff[M])=%d\n", total_time, diff[M]);
  return 0;
}

void test_mul() {
  loop_fission_3();
  loop_fission_4();
  return;
}

/*
$ gcc loop_fission.c -O3 -fopt-info

loop_fission.c:24:3: optimized:   Inlining printf/15 into loop_fission_1/5547 (always_inline).
loop_fission.c:46:3: optimized:   Inlining printf/15 into loop_fission_2/5548 (always_inline).
loop_fission.c:15:3: optimized: loop vectorized using 16 byte vectors
loop_fission.c:38:3: optimized: loop vectorized using 16 byte vectors
loop_fission.c:34:3: optimized: loop vectorized using 16 byte vectors

loop_fission_1 total_time: 0.000002 seconds vec[4]=516
loop_fission_2 total_time: 0.000001 seconds vec[4]=516
loop_fission_3 total_time: 0.000001 seconds
loop_fission_4 total_time: 0.000000 seconds
*/
int main() {
  test_add();
  test_mul();
  return 0;
}