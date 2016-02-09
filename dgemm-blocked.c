const char* dgemm_desc = "Simple blocked dgemm.";
#include "avxintrin-emu.h"

#if defined(BLOCK_SIZE)
#undef BLOCK_SIZE
#endif
#define BLOCK_SIZE 52
#define SMALL_BLOCK_1 8

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int i=0,j=0,k=0;
  double cij0,cij1,cij2,cij3;
  // double* result = (double*) malloc(sizeof(double)*2);
  /* For each row i of A */
  for (j = 0; j < N; ++j) 
  {
    /* For each column j of B */ 
    for (i = 0; i < M/4*4; i += 4)
    {
      /* Compute C(i,j) */
      // __mm256d Cvec = _mm256_loadu_pd(C + (i+j*lda));
      cij0 = C[i+j*lda];
      cij1 = C[i+1+j*lda];
      cij2 = C[i+2+j*lda];
      cij3 = C[i+3+j*lda];
      for (k=0; k < K/4*4; k+=4) {
      	// __mm256d Bvec = _mm256_loadu_pd(B + k+j*lda);
        cij0 += A[k+i*lda] * B[k+j*lda] + A[k+1+i*lda] * B[k+1+j*lda] + A[k+2+i*lda] * B[k+2+j*lda] + A[k+3+i*lda] * B[k+3+j*lda];
        cij1 += A[k+(i+1)*lda] * B[k+j*lda] + A[k+1+(i+1)*lda] * B[k+1+j*lda] + A[k+2+(i+1)*lda] * B[k+2+j*lda] + A[k+3+(i+1)*lda] * B[k+3+j*lda];
        cij2 += A[k+(i+2)*lda] * B[k+j*lda] + A[k+1+(i+2)*lda] * B[k+1+j*lda] + A[k+2+(i+2)*lda] * B[k+2+j*lda] + A[k+3+(i+2)*lda] * B[k+3+j*lda];
        cij3 += A[k+(i+3)*lda] * B[k+j*lda] + A[k+1+(i+3)*lda] * B[k+1+j*lda] + A[k+2+(i+3)*lda] * B[k+2+j*lda] + A[k+3+(i+3)*lda] * B[k+3+j*lda];
      }
      for (k=K/4*4; k < K; k++) {
        cij0 += A[k+i*lda] * B[k+j*lda];
        cij1 += A[k+(i+1)*lda] * B[k+j*lda];
        cij2 += A[k+(i+2)*lda] * B[k+j*lda];
        cij3 += A[k+(i+3)*lda] * B[k+j*lda];
      }
      C[i+j*lda] = cij0;
      C[i+1+j*lda] = cij1;
      C[i+2+j*lda] = cij2;
      C[i+3+j*lda] = cij3;
    }
        /* For each column j of B */ 
    for (i = M/4*4; i < M; i ++)
    {
      /* Compute C(i,j) */
      cij0 = C[i+j*lda];
      for (k=0; k < K/4*4; k+=4) {
        cij0 += A[k+i*lda] * B[k+j*lda] + A[k+1+i*lda] * B[k+1+j*lda] + A[k+2+i*lda] * B[k+2+j*lda] + A[k+3+i*lda] * B[k+3+j*lda];
      }
      for (k=K/4*4; k < K; k++) {
        cij0 += A[k+i*lda] * B[k+j*lda];
      }
      C[i+j*lda] = cij0;
    }
  }
  // free(result);
}

static void transpose(int lda, double *A, double *Atrans) {
  int j = 0;
  for (int i = 0; i < lda; i++) {
    for (j = 0; j < lda/4*4; j+=4) {
      Atrans[j+i*lda] = A[i+j*lda];
      Atrans[(j+1)+i*lda] = A[i+(j+1)*lda];
      Atrans[(j+2)+i*lda] = A[i+(j+2)*lda];
      Atrans[(j+3)+i*lda] = A[i+(j+3)*lda];
    }
    for (; j < lda; j++) {
      Atrans[j+i*lda] = A[i+j*lda];
    }
  }
}


void smallblock_dgemm (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += SMALL_BLOCK_1)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += SMALL_BLOCK_1)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += SMALL_BLOCK_1)
      {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (SMALL_BLOCK_1, lda-i);
          int N = min (SMALL_BLOCK_1, lda-j);
          int K = min (SMALL_BLOCK_1, lda-k);

          /* Perform individual block dgemm */
          do_block(lda, M, N, K, A + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  double *Atrans = malloc(sizeof(double)*lda*lda);
  transpose(lda, A, Atrans);
  /* For each block-row of A */ 
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (BLOCK_SIZE, lda-i);
          int N = min (BLOCK_SIZE, lda-j);
          int K = min (BLOCK_SIZE, lda-k);

          /* Perform individual block dgemm */
          smallblock_dgemm(lda, M, N, K, Atrans + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
  free(Atrans);
}



