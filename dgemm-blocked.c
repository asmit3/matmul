const char* dgemm_desc = "Simple blocked dgemm.";
#include "avxintrin-emu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(BLOCK_SIZE)
#undef BLOCK_SIZE
#endif
#define BLOCK_SIZE 48
#define SMALL_BLOCK_1 24


#define min(a,b) (((a)<(b))?(a):(b))

//static double Ablock[SMALL_BLOCK_1*SMALL_BLOCK_1];
//static double Bblock[BLOCK_SIZE*BLOCK_SIZE];

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int i=0,j=0,k=0;
  double cij00,cij10,cij20,cij30;
  double cij01,cij11,cij21,cij31;
  double cij02,cij12,cij22,cij32;
  double cij03,cij13,cij23,cij33;
  // double* result = (double*) malloc(sizeof(double)*2);
  /* For each row i of A */
  for (j = 0; j < N; j+=4) 
  {
    /* For each column j of B */ 
    for (i = 0; i < M; i += 4)
    {
      /* Compute C(i,j) */
      // __mm256d Cvec = _mm256_loadu_pd(C + (i+j*lda));
      cij00 = C[i+j*lda];
      cij10 = C[i+1+j*lda];
      cij20 = C[i+2+j*lda];
      cij30 = C[i+3+j*lda];

      cij01 = C[i+(j+1)*lda];
      cij11 = C[i+1+(j+1)*lda];
      cij21 = C[i+2+(j+1)*lda];
      cij31 = C[i+3+(j+1)*lda];

      cij02 = C[i+(j+2)*lda];
      cij12 = C[i+1+(j+2)*lda];
      cij22 = C[i+2+(j+2)*lda];
      cij32 = C[i+3+(j+2)*lda];   

      cij03 = C[i+(j+3)*lda];
      cij13 = C[i+1+(j+3)*lda];
      cij23 = C[i+2+(j+3)*lda];
      cij33 = C[i+3+(j+3)*lda];
      for (k=0; k < K; k+=4) {
      	// __mm256d Bvec = _mm256_loadu_pd(B + k+j*lda);
        cij00 += A[i+k*lda] * B[k+j*lda] + A[i+(k+1)*lda] * B[k+1+j*lda] + A[i+(k+2)*lda] * B[k+2+j*lda] + A[i+(k+3)*lda] * B[k+3+j*lda];
        cij10 += A[i+1+(k)*lda] * B[k+j*lda] + A[i+1+(k+1)*lda] * B[k+1+j*lda] + A[i+1+(k+2)*lda] * B[k+2+j*lda] + A[i+1+(k+3)*lda] * B[k+3+j*lda];
        cij20 += A[i+2+(k)*lda] * B[k+j*lda] + A[i+2+(k+1)*lda] * B[k+1+j*lda] + A[i+2+(k+2)*lda] * B[k+2+j*lda] + A[i+2+(k+3)*lda] * B[k+3+j*lda];
        cij30 += A[i+3+(k)*lda] * B[k+j*lda] + A[i+3+(k+1)*lda] * B[k+1+j*lda] + A[i+3+(k+2)*lda] * B[k+2+j*lda] + A[i+3+(k+3)*lda] * B[k+3+j*lda];
        
        cij01 += A[i+k*lda] * B[k+(j+1)*lda] + A[i+(k+1)*lda] * B[k+1+(j+1)*lda] + A[i+(k+2)*lda] * B[k+2+(j+1)*lda] + A[i+(k+3)*lda] * B[k+3+(j+1)*lda];
        cij11 += A[i+1+(k)*lda] * B[k+(j+1)*lda] + A[i+1+(k+1)*lda] * B[k+1+(j+1)*lda] + A[i+1+(k+2)*lda] * B[k+2+(j+1)*lda] + A[i+1+(k+3)*lda] * B[k+3+(j+1)*lda];
        cij21 += A[i+2+(k)*lda] * B[k+(j+1)*lda] + A[i+2+(k+1)*lda] * B[k+1+(j+1)*lda] + A[i+2+(k+2)*lda] * B[k+2+(j+1)*lda] + A[i+2+(k+3)*lda] * B[k+3+(j+1)*lda];
        cij31 += A[i+3+(k)*lda] * B[k+(j+1)*lda] + A[i+3+(k+1)*lda] * B[k+1+(j+1)*lda] + A[i+3+(k+2)*lda] * B[k+2+(j+1)*lda] + A[i+3+(k+3)*lda] * B[k+3+(j+1)*lda];
        
        cij02 += A[i+k*lda] * B[k+(j+2)*lda] + A[i+(k+1)*lda] * B[k+1+(j+2)*lda] + A[i+(k+2)*lda] * B[k+2+(j+2)*lda] + A[i+(k+3)*lda] * B[k+3+(j+2)*lda];
        cij12 += A[i+1+(k)*lda] * B[k+(j+2)*lda] + A[i+1+(k+1)*lda] * B[k+1+(j+2)*lda] + A[i+1+(k+2)*lda] * B[k+2+(j+2)*lda] + A[i+1+(k+3)*lda] * B[k+3+(j+2)*lda];
        cij22 += A[i+2+(k)*lda] * B[k+(j+2)*lda] + A[i+2+(k+1)*lda] * B[k+1+(j+2)*lda] + A[i+2+(k+2)*lda] * B[k+2+(j+2)*lda] + A[i+2+(k+3)*lda] * B[k+3+(j+2)*lda];
        cij32 += A[i+3+(k)*lda] * B[k+(j+2)*lda] + A[i+3+(k+1)*lda] * B[k+1+(j+2)*lda] + A[i+3+(k+2)*lda] * B[k+2+(j+2)*lda] + A[i+3+(k+3)*lda] * B[k+3+(j+2)*lda];
        
        cij03 += A[i+k*lda] * B[k+(j+3)*lda] + A[i+(k+1)*lda] * B[k+1+(j+3)*lda] + A[i+(k+2)*lda] * B[k+2+(j+3)*lda] + A[i+(k+3)*lda] * B[k+3+(j+3)*lda];
        cij13 += A[i+1+(k)*lda] * B[k+(j+3)*lda] + A[i+1+(k+1)*lda] * B[k+1+(j+3)*lda] + A[i+1+(k+2)*lda] * B[k+2+(j+3)*lda] + A[i+1+(k+3)*lda] * B[k+3+(j+3)*lda];
        cij23 += A[i+2+(k)*lda] * B[k+(j+3)*lda] + A[i+2+(k+1)*lda] * B[k+1+(j+3)*lda] + A[i+2+(k+2)*lda] * B[k+2+(j+3)*lda] + A[i+2+(k+3)*lda] * B[k+3+(j+3)*lda];
        cij33 += A[i+3+(k)*lda] * B[k+(j+3)*lda] + A[i+3+(k+1)*lda] * B[k+1+(j+3)*lda] + A[i+3+(k+2)*lda] * B[k+2+(j+3)*lda] + A[i+3+(k+3)*lda] * B[k+3+(j+3)*lda];
      }
      C[i+j*lda] = cij00;
      C[i+1+j*lda] = cij10;
      C[i+2+j*lda] = cij20;
      C[i+3+j*lda] = cij30;

      C[i+(j+1)*lda] = cij01;
      C[i+1+(j+1)*lda] = cij11;
      C[i+2+(j+1)*lda] = cij21;
      C[i+3+(j+1)*lda] = cij31;

      C[i+(j+2)*lda] = cij02;
      C[i+1+(j+2)*lda] = cij12;
      C[i+2+(j+2)*lda] = cij22;
      C[i+3+(j+2)*lda] = cij32;

      C[i+(j+3)*lda] = cij03;
      C[i+1+(j+3)*lda] = cij13;
      C[i+2+(j+3)*lda] = cij23;
      C[i+3+(j+3)*lda] = cij33;

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


static void smallblock_dgemm (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* Accumulate block dgemms into block of C */
  for (int k = 0; k < K; k += SMALL_BLOCK_1)
  {
    /* For each block-row of A */
    for (int i = 0; i < M; i += SMALL_BLOCK_1)
    {
//      memcpy(Ablock, A + i + k*lda, sizeof(double)*SMALL_BLOCK_1*SMALL_BLOCK_1);
      /* For each block-column of B */
      for (int j = 0; j < N; j += SMALL_BLOCK_1)
      {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M1 = min (SMALL_BLOCK_1, M-i);
          int N1 = min (SMALL_BLOCK_1, N-j);
          int K1 = min (SMALL_BLOCK_1, K-k);

          /* Perform individual block dgemm */
          do_block(lda, M1, N1, K1, A+i+k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  // static double *Ablock = malloc(sizeof(double)*SMALL_BLOCK_1*SMALL_BLOCK_1);
//  double *Atrans = malloc(sizeof(double)*lda*lda);
//  transpose(lda, A, Atrans);
  int newsize = ((lda+7)/8)*8;
  //printf("%d\n",newsize); 


  double *Aalign, *Balign, *Calign;
  posix_memalign(&Aalign, sizeof(double)*8, sizeof(double)*newsize*newsize);
  posix_memalign(&Balign, sizeof(double)*8, sizeof(double)*newsize*newsize);
  posix_memalign(&Calign, sizeof(double)*8, sizeof(double)*newsize*newsize);
  // double *Aalign = malloc(sizeof(double)*newsize*newsize);
  // double *Balign = malloc(sizeof(double)*newsize*newsize);
  // double *Calign = malloc(sizeof(double)*newsize*newsize);



  for (int i = 0; i < lda; ++i)
  {
       memcpy(Aalign+i*newsize, A+i*lda, sizeof(double)*lda);
       memcpy(Balign+i*newsize, B+i*lda, sizeof(double)*lda);
       memset(Aalign+lda+i*newsize, 0.0, sizeof(double)*(newsize-lda));
       memset(Balign+lda+i*newsize, 0.0, sizeof(double)*(newsize-lda));

  }
  for (int i = lda; i < newsize; ++i)
  {
     memset(Aalign+i*newsize, 0.0, sizeof(double)*newsize);
     memset(Balign+i*newsize, 0.0, sizeof(double)*newsize);
  }
  memset(Calign, 0.0, sizeof(double)*newsize*newsize);
  // for (int i = 0; i < newsize; i++) {
  // 	for (int j = 0; j < newsize; j++) {
  // 		if (i >= lda || j >= lda) {
	 //  		Aalign[j+i*newsize] = 0;
	 //  		Balign[j+i*newsize] = 0;
	 //  	} else {
  // 	  		Aalign[j+i*newsize] = A[j+i*lda];
  // 	  		Balign[j+i*newsize] = B[j+i*lda];
  // 	  	}
  //     Calign[j+i*newsize] = 0;
  // 	}
  // }





  /* For each block-column of B */
  for (int j = 0; j < newsize; j += BLOCK_SIZE)
  {
    /* Accumulate block dgemms into block of C */
    for (int i = 0; i < newsize; i += BLOCK_SIZE)
    {
      /* For each block-row of A */ 
      for (int k = 0; k < newsize; k += BLOCK_SIZE)
      {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (BLOCK_SIZE, newsize-i);
          int N = min (BLOCK_SIZE, newsize-j);
          int K = min (BLOCK_SIZE, newsize-k);
          /* Perform individual block dgemm */
//           smallblock_dgemm(newsize, M, N, K, A + i + k*newsize, B + k + j*newsize, C + i + j*newsize);
          do_block(newsize, M, N, K, Aalign + i + k*newsize, Balign+k+j*newsize, Calign + i + j*newsize);
      }
    }
  }

  for (int i = 0; i < lda; ++i)
  {
     memcpy(C+i*lda, Calign+i*newsize, sizeof(double)*lda);
  }
  // for (int i = 0; i < lda; i++) {
  // 	for (int j = 0; j < lda; j++) {
  // 		C[j+i*lda] = Calign[j+i*newsize];
  // 	}
  // }

  // printf("HERE IS C correct\n");
  // for (int i = 0; i < lda; i++) {
  //   for (int j = 0; j < lda; j++) {
  //     printf("%+4.3f ", C[i + j*lda]);
  //   }
  //   printf("\n");
  // }
  // printf("HERE IS Calign\n");
  // for (int i = 0; i < newsize; i++) {
  //   for (int j = 0; j < newsize; j++) {
  //     printf("%+4.3f ", Calign[i + j*newsize]);
  //   }
  //   printf("\n");
  // }
  // C[0] = 1976342;

  free(Aalign);
  free(Balign);
  free(Calign);
}



