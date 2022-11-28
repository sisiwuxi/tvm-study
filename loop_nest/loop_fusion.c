#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#define N 51200
int test_handle_loop_fusion() {
  // clang loop_fusion.c
  int i, a[N], b[N], x[N], y[N];
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    a[i] = rand()%100;
    b[i] = rand()%100;
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++)
    x[i] = a[i] + b[i];
  for(i=0; i<N; i++)
    y[i] = a[i] - b[i];
  gettimeofday(&time_end, NULL);
  printf("before used time %ld us\n", time_end.tv_usec - time_start.tv_usec);

  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++)
    x[i] = a[i] + b[i];
    y[i] = a[i] - b[i];
  gettimeofday(&time_end, NULL);
  printf("unroll used time %ld us\n", time_end.tv_usec - time_start.tv_usec);

  printf("%d\n", x[4]);
  printf("%d\n", y[3]);
  return 0;
}

void test_loop_fusion() {
  // RUN: opt -S -loop-simplify -loop-fusion < %s | FileCheck %s
  // opt -S -loop-fusion loop_fusion.ll &> loop_fusion_opt.ll
  return;
}

int main() {
  test_handle_loop_fusion();
  test_loop_fusion();
  return 0;
}