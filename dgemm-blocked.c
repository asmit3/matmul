const char* dgemm_desc = "Simple blocked dgemm.";
#include "avxintrin-emu.h"

#if defined(BLOCK_SIZE)
#undef BLOCK_SIZE
#endif
#define BLOCK_SIZE 48

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int i=0,j=0,k=0;
  double cij;
  double* result = (double*) malloc(sizeof(double)*2);
  /* For each row i of A */
  for (i = 0; i < M; ++i)
  {
    for (k=0; k< (K/4)*4; k+=4)
    /* For each column j of B */
    {
      __m256d Avec = _mm256_loadu_pd(A + (k+i*lda));
//      __m256d Cvec = _mm256_loadu_pd(A + (k+4+i*lda));
      for (j = 0; j < N; ++j) 
      {
      /* Compute C(i,j) */
        cij = C[i+j*lda];
//        __m256d Avec = _mm256_loadu_pd(A + (k+i*lda));
        __m256d Bvec = _mm256_loadu_pd(B + (k+j*lda));
//        __m256d Cvec = _mm256_loadu_pd(A + (k+4+i*lda));
//        __m256d Dvec = _mm256_loadu_pd(B + (k+4+j*lda));
        __m256d prod1 = _mm256_mul_pd(Avec, Bvec);
//        __m256d prod2 = _mm256_mul_pd(Cvec, Dvec);
        __m256d temp = _mm256_hadd_pd(prod1, prod1);
//        __m128d dotproduct = _mm_add_pd( _mm256_extractf128_pd( temp, 0 ), _mm256_extractf128_pd( temp, 1 ) );
        _mm_storeu_pd(result, _mm256_extractf128_pd(temp,0));
        cij += result[0] + result[1];
//        for (k=(K/8)*8; k < K; ++k)
//        {
//          cij += A[k+i*lda] * B[k+j*lda];
//        }
        C[i+j*lda] = cij;
      }
    }
    for (k=(K/4)*4; k<K; ++k)
    {
      for (j=0; j< N; ++j)
      {
        
        
        C[i+j*lda] += A[k+i*lda]*B[k+j*lda];
      }
    }
//    C[i+j*lda] = cij;
  }
  free(result);
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
          do_block(lda, M, N, K, Atrans + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
  free(Atrans);
}

