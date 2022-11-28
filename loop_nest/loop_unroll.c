#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define VECTOR_ENABLE 1

void test_original() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j]   = A[i][j] + B[i][j]*C[i][j];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("test_original used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_inner_2() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++){
    for(j=0; j<N; j+=2){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j];
      A[i][j+1] = A[i][j+1] + B[i][j+1]*C[i][j+1];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("test_unroll_inner_2 used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_inner_4() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++){
    for(j=0; j<N; j+=4){
      A[i][j]     = A[i][j]     + B[i][j]*C[i][j];
      A[i][j+1]   = A[i][j+1]   + B[i][j+1]*C[i][j+1];
      A[i][j+2]   = A[i][j+2]   + B[i][j+2]*C[i][j+2];
      A[i][j+3]   = A[i][j+3]   + B[i][j+3]*C[i][j+3];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("test_unroll_inner_4 used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_inner_8() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++){
    for(j=0; j<N; j+=8){
      A[i][j]     = A[i][j]     + B[i][j]*C[i][j];
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
  printf("test_unroll_inner_8 used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_inner_16() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++){
    for(j=0; j<N; j+=16){
      A[i][j]     = A[i][j]     + B[i][j]*C[i][j];
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
  printf("test_unroll_inner_16 used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_unroll_outer_2() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = j;
      B[i][j] = j;
      C[i][j] = j;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i+=2){
    for(j=0; j<N; j++){
      A[i][j]   = A[i][j]   + B[i][j]*C[i][j];
      A[i+1][j] = A[i+1][j] + B[i+1][j]*C[i+1][j];
    }
  }
  gettimeofday(&time_end, NULL);
  printf("test_unroll_outer_2 used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
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
  // clang loop_unroll.c -O1 -Rpass=loop-unroll
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


void test_pragma_scanf(){
  // clang loop_unroll.c -O1 -Rpass=loop-unroll
  int i,N,M,sum;
  N = 1024;
  sum = 0;
  int A[N], B[N];
  for(i=0; i<N; i++){
    A[i] = i;
    B[i] = rand()%10;
  }
  printf("Input(<512):");
  scanf("%d", &M);
// #pragma clang loop unroll(enable)
#pragma clang loop unroll(full)
// #pragma clang loop unroll_count(8)
  for(i=0; i<M; i++){
    sum = sum + A[i] + B[i];
  }
  printf("output = %d\n", sum);
}

#if VECTOR_ENABLE == 1
void test_vector_1() {
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j,k;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = 1.0;
      B[i][j] = 2.0;
      C[i][j] = 3.0;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = A[i][j] + B[i][j]*C[i][j];
    }
  }
  // for(i=1; i<N; i++){
  //   for(j=1; j<N; j++){
  //     printf("%f\n", A[i][j]);
  //   }
  // }
  gettimeofday(&time_end, NULL);
  printf("test_vector_1 used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

// #include<x86intrin.h>
#include<immintrin.h>
void test_vector_2() {
  register __m256d ymm3, ymm4, ymm5;
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j,k;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = 1.0;
      B[i][j] = 2.0;
      C[i][j] = 3.0;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++){
    for(j=0; j<N; j+=4){
      ymm3 = _mm256_load_pd(A[i]+j);
      ymm4 = _mm256_load_pd(B[i]+j);
      ymm5 = _mm256_load_pd(C[i]+j);
      ymm4 = _mm256_mul_pd(ymm4,ymm5);
      ymm3 = _mm256_add_pd(ymm3,ymm4);
      _mm256_store_pd((A[i]+j),ymm3);
    }
  }
  gettimeofday(&time_end, NULL);
  printf("test_vector_2 used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
}

void test_vector_3() {
  register __m256d ymm3, ymm4, ymm5, ymm6;
  const int N=512;
  double A[N][N], B[N][N], C[N][N];
  int i,j,k;
  struct timeval time_start, time_end;
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      A[i][j] = 1.0;
      B[i][j] = 2.0;
      C[i][j] = 3.0;
    }
  }
  gettimeofday(&time_start, NULL);
  for(i=0; i<N; i++){
    for(j=0; j<N; j+=8){ // 4*2
      ymm3 = _mm256_load_pd(A[i]+j);
      ymm4 = _mm256_load_pd(B[i]+j);
      ymm5 = _mm256_load_pd(C[i]+j);
      ymm4 = _mm256_mul_pd(ymm4,ymm5);
      ymm6 = _mm256_add_pd(ymm3,ymm4); // add ymm6
      _mm256_store_pd((A[i]+j), ymm6);
      ymm3 = _mm256_load_pd(A[i]+j+4);
      ymm4 = _mm256_load_pd(B[i]+j+4);
      ymm5 = _mm256_load_pd(C[i]+j+4);
      ymm4 = _mm256_mul_pd(ymm4,ymm5);
      ymm6 = _mm256_add_pd(ymm3,ymm4);
      _mm256_store_pd((A[i]+j+4), ymm6);
    }
  }
  gettimeofday(&time_end, NULL);
  printf("test_vector_3 used time %ld us\n", time_end.tv_usec - time_start.tv_usec);
} 

void test_vector(){
  // clang loop_unroll.c -march=haswell -O3
  test_vector_1();
  test_vector_2();
  test_vector_3();
}
#endif

int main() {
  // test_original();
  // test_unroll_inner_2();
  // test_unroll_inner_4();
  // test_unroll_inner_8();
  // test_unroll_inner_16();
  // test_unroll_outer_2();
  // test_llvm_unroll_loops();
  // test_pragma();
  #if VECTOR_ENABLE == 1
  test_vector();
  #endif
}