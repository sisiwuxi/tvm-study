/*
1
The most time-consuming program is generally presented as a loop form

2
Loop unrolling
  Reduce the number of branch instruction
  Increase the space for processor instruction scheduling
  Get more opportunities for instruction parallelism
  Increase the rate of register reuse
Loop compression
2.1
before:
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = A[i][j] + B[i][j]*C[i][j]
    }
  }
after:
  for(i=0; i<N; i++){
    for(j=0; j<N; j+=4){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j]
      A[i][j+1] = A[i][j+1] + B[i][j+1]*C[i][j+1]
      A[i][j+2] = A[i][j+2] + B[i][j+2]*C[i][j+2]
      A[i][j+3] = A[i][j+3] + B[i][j+3]*C[i][j+3]
    }
  }

3
Illegality
3.1
before:
  for(i=1; i<N; i++){
    for(j=1; j<N; j++){
      A[i+1][j-1] = A[i][j] + 3
    }
  }
  dependent
      j=1             j=2             j=3             j=4
  i=1 A[2][0]=A[1][1] A[2][1]=A[1][2] A[2][2]=A[1][3] A[2][3]=A[1][4]
  i=2 A[3][0]=A[2][1] A[3][1]=A[2][2] A[3][2]=A[2][3] A[3][3]=A[2][4]
  i=3 A[4][0]=A[3][1] A[4][1]=A[3][2] A[4][2]=A[3][3] A[4][3]=A[3][4]
  i=4 A[5][0]=A[4][1] A[5][1]=A[4][2] A[5][2]=A[4][3] A[5][3]=A[4][4]

after:
  for(i=1; i<N; i+=2){
    for(j=1; j<N; j++){
      A[i+1][j-1] = A[i][j] + 3
      A[i+2][j-1] = A[i+1][j] + 3
    }
  }
  dependent
      j=1             j=2             j=3             j=4
  i=1 A[2][0]=A[1][1] A[2][1]=A[1][2] A[2][2]=A[1][3] A[2][3]=A[1][4]
      A[3][0]=A[2][1] A[3][1]=A[2][2] A[3][2]=A[2][3] A[3][3]=A[2][4]
  i=3 A[4][0]=A[3][1] A[4][1]=A[3][2] A[4][2]=A[3][3] A[4][3]=A[3][4]
      A[5][0]=A[4][1] A[5][1]=A[4][2] A[5][2]=A[4][3] A[5][3]=A[4][4]
  
  problem:
  A[3][0]=A[2][1]
    before: A[2][1]=A[1][2], ...
    after:  A[2][1] not update

4
Fully loop unroll
4.1
before
void loop_unroll(void) {
  float a[8];
  for(int i=0; i<8; i++) {
    a[i] = a[i] + 3;
  }
}

after
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
Unroll is not the more the better
Need more register, Increase the risk of instruction buffer overflow

5
Not evenly divisible
before
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
after
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

6
llvm
  -funroll-loops
  -fno-unroll-loops
  -mllvm -unroll-max-count
  -mllvm -unroll-count
  -mllvm -unroll-runtime
  -mllvm -unroll-threshold
  -mllvm -unroll-remainder
  -Rpass=loop-unroll
  -Rpass-missed=loop-unroll
progma
  #pragma clang loop unroll(enable)
  #pragma clang loop unroll(full)
  #pragma clang loop unroll_count(8)
*/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>


void test_original() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=1; i<N; i++){
    for(j=1; j<N; j++){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=1; i<N; i++){
    for(j=1; j<N; j++){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("unroll used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_inner_2() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=1; i<N; i++){
    for(j=1; j<N; j+=2){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
      A[i][j+1] = j+1;
      B[i][j+1] = j+1;
      C[i][j+1] = j+1;      
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=1; i<N; i++){
    for(j=1; j<N; j+=2){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j];
      A[i][j+1]   = A[i][j+1]   + B[i][j+1]*C[i][j+1];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("unroll used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_inner_4() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=1; i<N; i++){
    for(j=1; j<N; j+=4){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
      A[i][j+1] = j+1;
      B[i][j+1] = j+1;
      C[i][j+1] = j+1; 
      A[i][j+2] = j+2;
      B[i][j+2] = j+2;
      C[i][j+2] = j+2;
      A[i][j+3] = j+3;
      B[i][j+3] = j+3;
      C[i][j+3] = j+3;                 
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=1; i<N; i++){
    for(j=1; j<N; j+=4){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j];
      A[i][j+1]   = A[i][j+1]   + B[i][j+1]*C[i][j+1];
      A[i][j+2]   = A[i][j+2]   + B[i][j+2]*C[i][j+2];
      A[i][j+3]   = A[i][j+3]   + B[i][j+3]*C[i][j+3];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("unroll used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_inner_8() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=1; i<N; i++){
    for(j=1; j<N; j+=8){
      A[i][j] = j;      B[i][j] = j;      C[i][j] = j;
      A[i][j+1] = j+1;  B[i][j+1] = j+1;  C[i][j+1] = j+1; 
      A[i][j+2] = j+2;  B[i][j+2] = j+2;  C[i][j+2] = j+2;
      A[i][j+3] = j+3;  B[i][j+3] = j+3;  C[i][j+3] = j+3; 
      A[i][j+4] = j+4;  B[i][j+4] = j+4;  C[i][j+4] = j+4; 
      A[i][j+5] = j+5;  B[i][j+5] = j+5;  C[i][j+5] = j+5;
      A[i][j+6] = j+6;  B[i][j+6] = j+6;  C[i][j+6] = j+6; 
      A[i][j+7] = j+7;  B[i][j+7] = j+7;  C[i][j+7] = j+7; 
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=1; i<N; i++){
    for(j=1; j<N; j+=8){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j];
      A[i][j+1]   = A[i][j+1]   + B[i][j+1]*C[i][j+1];
      A[i][j+2]   = A[i][j+2]   + B[i][j+2]*C[i][j+2];
      A[i][j+3]   = A[i][j+3]   + B[i][j+3]*C[i][j+3];
      A[i][j+4]   = A[i][j+4]   + B[i][j+4]*C[i][j+4];
      A[i][j+5]   = A[i][j+5]   + B[i][j+5]*C[i][j+5];
      A[i][j+6]   = A[i][j+6]   + B[i][j+6]*C[i][j+6];
      A[i][j+7]   = A[i][j+7]   + B[i][j+7]*C[i][j+7];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("unroll used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_inner_16() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=1; i<N; i++){
    for(j=1; j<N; j+=16){
      A[i][j] = j;      B[i][j] = j;      C[i][j] = j;
      A[i][j+1] = j+1;  B[i][j+1] = j+1;  C[i][j+1] = j+1; 
      A[i][j+2] = j+2;  B[i][j+2] = j+2;  C[i][j+2] = j+2;
      A[i][j+3] = j+3;  B[i][j+3] = j+3;  C[i][j+3] = j+3; 
      A[i][j+4] = j+4;  B[i][j+4] = j+4;  C[i][j+4] = j+4; 
      A[i][j+5] = j+5;  B[i][j+5] = j+5;  C[i][j+5] = j+5;
      A[i][j+6] = j+6;  B[i][j+6] = j+6;  C[i][j+6] = j+6; 
      A[i][j+7] = j+7;  B[i][j+7] = j+7;  C[i][j+7] = j+7;
      A[i][j+8] = j+8;  B[i][j+1] = j+8;  C[i][j+1] = j+8; 
      A[i][j+9] = j+9;  B[i][j+2] = j+9;  C[i][j+2] = j+9;
      A[i][j+10] = j+10;  B[i][j+3] = j+10;  C[i][j+3] = j+10; 
      A[i][j+11] = j+11;  B[i][j+4] = j+11;  C[i][j+4] = j+11; 
      A[i][j+12] = j+12;  B[i][j+5] = j+12;  C[i][j+5] = j+12;
      A[i][j+13] = j+13;  B[i][j+6] = j+13;  C[i][j+6] = j+13; 
      A[i][j+14] = j+14;  B[i][j+7] = j+14;  C[i][j+7] = j+14;       
      A[i][j+15] = j+15;  B[i][j+7] = j+15;  C[i][j+7] = j+15;       
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=1; i<N; i++){
    for(j=1; j<N; j+=16){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j];
      A[i][j+1]   = A[i][j+1]   + B[i][j+1]*C[i][j+1];
      A[i][j+2]   = A[i][j+2]   + B[i][j+2]*C[i][j+2];
      A[i][j+3]   = A[i][j+3]   + B[i][j+3]*C[i][j+3];
      A[i][j+4]   = A[i][j+4]   + B[i][j+4]*C[i][j+4];
      A[i][j+5]   = A[i][j+5]   + B[i][j+5]*C[i][j+5];
      A[i][j+6]   = A[i][j+6]   + B[i][j+6]*C[i][j+6];
      A[i][j+7]   = A[i][j+7]   + B[i][j+7]*C[i][j+7];
      A[i][j+8]   = A[i][j+8]   + B[i][j+8]*C[i][j+8];
      A[i][j+9]   = A[i][j+9]   + B[i][j+9]*C[i][j+9];
      A[i][j+10]   = A[i][j+10]   + B[i][j+10]*C[i][j+10];
      A[i][j+11]   = A[i][j+11]   + B[i][j+11]*C[i][j+11];
      A[i][j+12]   = A[i][j+12]   + B[i][j+12]*C[i][j+12];
      A[i][j+13]   = A[i][j+13]   + B[i][j+13]*C[i][j+13];
      A[i][j+14]   = A[i][j+14]   + B[i][j+14]*C[i][j+14];
      A[i][j+15]   = A[i][j+15]   + B[i][j+15]*C[i][j+15];      
    }
  }
  gettimeofday(&time_end, NULL);
  printf("unroll used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_outer_2() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=1; i<N; i+=2){
    for(j=1; j<N; j++){
      A[i][j] = j;      B[i][j] = j;      C[i][j] = j;
      A[i+1][j] = j;      B[i+1][j] = j;      C[i+1][j] = j;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=1; i<N; i+=2){
    for(j=1; j<N; j++){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j];
      A[i+1][j]   = A[i+1][j]   + B[i+1][j]*C[i+1][j];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("unroll used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

// # define N 128
void test_llvm_unroll_loops() {
  // clang loop_unroll.c -O1 -funroll-loops -emit-llvm -S -Rpass=loop-unroll --std=c90
  const int N=128;
  float sum = 0;
  float a[N];
  int i,j;
  for(i=0; i<N; i++){
    a[i] = i;
  }
  // %38 = fadd double %32, %37, !dbg !38
  for(j=0; j<N; j++){
    sum = sum + a[j];
  }
  printf("sum = %f", sum);
}

void test_pragma(){
  int i,N,sum;
  N = 1024;
  sum = 0;
  int A[N], B[N];
  for(i=0; i<N; i++){
    A[i] = i;
    B[i] = rand()%10;
  }
#pragma clang loop unroll(enable)
// #pragma clang loop unroll(full)
// #pragma clang loop unroll_count(8)
  for(i=0; i<N; i++){
    sum = sum + A[i] + B[i];
  }
  printf("%d\n", sum);
}

int main() {
  // test_original();
  // test_unroll_inner_2();
  // test_unroll_inner_4();
  // test_unroll_inner_8();
  // test_unroll_inner_16();
  // test_llvm_unroll_loops();
  test_pragma();
}