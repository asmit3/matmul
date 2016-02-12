const char* dgemm_desc = "Naive, three-loop dgemm.";
#include <stdlib.h>
#define min(a,b) (((a)<(b))?(a):(b))

static inline void transpose(int lda, double *A, double *Atrans) {
  int j = 0;
  for (int i = 0; i < lda; i++) {
    for (j = 0; j < lda; j++) {
      Atrans[j+i*lda] = A[i+j*lda];
    }
  }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_dgemm (int n, double* A, double* B, double* C)
{
  int blocksize = 16;
  double *Atrans = alloca(sizeof(double)*n*n);
  transpose(n, A, Atrans);
  for (int i_block = 0; i_block < n; i_block += blocksize)
  {
    for (int j_block = 0; j_block < n; j_block += blocksize)
    {
      /* For each column j of B */
      for (int j = j_block; j < min(j_block+blocksize,n); ++j) 
      {
        /* For each row i of A */
        for (int i = i_block; i < min(i_block+blocksize,n); ++i)
        {
          /* Compute C(i,j) */
          double cij = C[i+j*n];
          for( int k = 0; k < n; ++k )
          {
            cij += Atrans[k+i*n] * B[k+j*n];
          }
          C[i+j*n] = cij;
        }
      }
    }
  }
}
