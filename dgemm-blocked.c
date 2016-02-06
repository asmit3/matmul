const char* dgemm_desc = "Simple blocked dgemm.";
// #include "avxintrin_emu.h"

#if defined(BLOCK_SIZE)
#undef BLOCK_SIZE
#endif
#define BLOCK_SIZE 50

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int i=0,j=0,k=0;
  // double fa0,fb0,fa1,fb1,fa2,fb2,fa3,fb3;
  double *Bvec;
  double *Avec;
  double cij;
  /* For each row i of A */
  for (i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      cij = C[i+j*lda];
      for (k = 0; k < (K/4)*4; k+=4) {
        Avec = A + (k+i*lda);
        Bvec = B + (k+j*lda);
        cij += Avec[0]*Bvec[0] + Avec[1]*Bvec[1] + Avec[2]*Bvec[2] + Avec[3]*Bvec[3];
      }
      for (k=(K/4)*4; k < K; ++k) {
        cij += A[k+i*lda] * B[k+j*lda];
      }
      C[i+j*lda] = cij;
    }
}

static void transpose(int lda, double *A, double *Atrans) {
  for (int i = 0; i < lda; i++) {
    for (int j = 0; j < lda; j++) {
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
}

