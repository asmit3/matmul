const char* dgemm_desc = "Simple blocked dgemm.";
#include "avxintrin-emu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(BLOCK_SIZE)
#undef BLOCK_SIZE
#endif
#define BLOCK_SIZE 1024
#define SMALL_BLOCK_1 24


#define I_BLOCK 1024
#define J_BLOCK 32
#define K_BLOCK 32

#define min(a,b) (((a)<(b))?(a):(b))

//static double Ablock[SMALL_BLOCK_1*SMALL_BLOCK_1];
//static double Bblock[BLOCK_SIZE*BLOCK_SIZE];

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int i=0,j=0,k=0;
  // double cij00,cij10,cij20,cij30;
  // double cij01,cij11,cij21,cij31;
  // double cij02,cij12,cij22,cij32;
  // double cij03,cij13,cij23,cij33;
  __m256d Bvec00,Bvec10,Bvec20,Bvec30,Bvec01,Bvec11,Bvec21,Bvec31,Bvec02,Bvec12,Bvec22,Bvec32,Bvec03,Bvec13,Bvec23,Bvec33,Avec0,Avec1,Avec2,Avec3,Cvec0,Cvec1,Cvec2,Cvec3;
  // __m256d Bvec00, Bvec10, Bvec01, Bvec11, Bvec02, Bvec12, Bvec03, Bvec13,Avec0,Avec1,Avec2,Avec3,Cvec0,Cvec1,Cvec2,Cvec3;
  // double* result = (double*) malloc(sizeof(double)*2);
  /* For each row i of A */
  for (j = 0; j < N; j+=4) 
  {
  /* For each column j of B */ 
    for (k=0; k < K; k+=4) {
      // cij00 = C[i+j*lda];
      // cij10 = C[i+1+j*lda];
      // cij20 = C[i+2+j*lda];
      // cij30 = C[i+3+j*lda];

      // cij01 = C[i+(j+1)*lda];
      // cij11 = C[i+1+(j+1)*lda];
      // cij21 = C[i+2+(j+1)*lda];
      // cij31 = C[i+3+(j+1)*lda];

      // cij02 = C[i+(j+2)*lda];
      // cij12 = C[i+1+(j+2)*lda];
      // cij22 = C[i+2+(j+2)*lda];
      // cij32 = C[i+3+(j+2)*lda];   

      // cij03 = C[i+(j+3)*lda];
      // cij13 = C[i+1+(j+3)*lda];
      // cij23 = C[i+2+(j+3)*lda];
      // cij33 = C[i+3+(j+3)*lda];

      Bvec00 = _mm256_broadcast_sd(B+k+j*lda);
      Bvec10 = _mm256_broadcast_sd(B+k+(j+1)*lda);
      Bvec20 = _mm256_broadcast_sd(B+k+(j+2)*lda);
      Bvec30 = _mm256_broadcast_sd(B+k+(j+3)*lda);

      Bvec01 = _mm256_broadcast_sd(B+k+1+j*lda);
      Bvec11 = _mm256_broadcast_sd(B+k+1+(j+1)*lda);
      Bvec21 = _mm256_broadcast_sd(B+k+1+(j+2)*lda);
      Bvec31 = _mm256_broadcast_sd(B+k+1+(j+3)*lda);

      Bvec02 = _mm256_broadcast_sd(B+k+2+j*lda);
      Bvec12 = _mm256_broadcast_sd(B+k+2+(j+1)*lda);
      Bvec22 = _mm256_broadcast_sd(B+k+2+(j+2)*lda);
      Bvec32 = _mm256_broadcast_sd(B+k+2+(j+3)*lda);

      Bvec03 = _mm256_broadcast_sd(B+k+3+j*lda);
      Bvec13 = _mm256_broadcast_sd(B+k+3+(j+1)*lda);
      Bvec23 = _mm256_broadcast_sd(B+k+3+(j+2)*lda);
      Bvec33 = _mm256_broadcast_sd(B+k+3+(j+3)*lda);

      for (i = 0; i < M; i += 4)
      {

        /* Compute C(i,j) */
        Cvec0 = _mm256_load_pd(C + (i+j*lda));
        Cvec1 = _mm256_load_pd(C + (i+(j+1)*lda));
        Cvec2 = _mm256_load_pd(C + (i+(j+2)*lda));
        Cvec3 = _mm256_load_pd(C + (i+(j+3)*lda));

      	Avec0 = _mm256_load_pd(A + i+k*lda);
        Avec1 = _mm256_load_pd(A + i+(k+1)*lda);
        Avec2 = _mm256_load_pd(A + i+(k+2)*lda);
        Avec3 = _mm256_load_pd(A + i+(k+3)*lda);

        Cvec0 = _mm256_add_pd(Cvec0, _mm256_mul_pd(Avec0, Bvec00));
        Cvec0 = _mm256_add_pd(Cvec0, _mm256_mul_pd(Avec1, Bvec01));
        Cvec0 = _mm256_add_pd(Cvec0, _mm256_mul_pd(Avec2, Bvec02));
        Cvec0 = _mm256_add_pd(Cvec0, _mm256_mul_pd(Avec3, Bvec03));

        Cvec1 = _mm256_add_pd(Cvec1, _mm256_mul_pd(Avec0, Bvec10));
        Cvec1 = _mm256_add_pd(Cvec1, _mm256_mul_pd(Avec1, Bvec11));
        Cvec1 = _mm256_add_pd(Cvec1, _mm256_mul_pd(Avec2, Bvec12));
        Cvec1 = _mm256_add_pd(Cvec1, _mm256_mul_pd(Avec3, Bvec13));

        Cvec2 = _mm256_add_pd(Cvec2, _mm256_mul_pd(Avec0, Bvec20));
        Cvec2 = _mm256_add_pd(Cvec2, _mm256_mul_pd(Avec1, Bvec21));
        Cvec2 = _mm256_add_pd(Cvec2, _mm256_mul_pd(Avec2, Bvec22));
        Cvec2 = _mm256_add_pd(Cvec2, _mm256_mul_pd(Avec3, Bvec23));

        Cvec3 = _mm256_add_pd(Cvec3, _mm256_mul_pd(Avec0, Bvec30));
        Cvec3 = _mm256_add_pd(Cvec3, _mm256_mul_pd(Avec1, Bvec31));
        Cvec3 = _mm256_add_pd(Cvec3, _mm256_mul_pd(Avec2, Bvec32));
        Cvec3 = _mm256_add_pd(Cvec3, _mm256_mul_pd(Avec3, Bvec33));


        _mm256_store_pd(C+i+j*lda, Cvec0);
        _mm256_store_pd(C+i+(j+1)*lda, Cvec1);
        _mm256_store_pd(C+i+(j+2)*lda, Cvec2);
        _mm256_store_pd(C+i+(j+3)*lda, Cvec3);

        // cij00 += A[i+k*lda] * B[k+j*lda] + A[i+(k+1)*lda] * B[k+1+j*lda] + A[i+(k+2)*lda] * B[k+2+j*lda] + A[i+(k+3)*lda] * B[k+3+j*lda];
        // cij10 += A[i+1+(k)*lda] * B[k+j*lda] + A[i+1+(k+1)*lda] * B[k+1+j*lda] + A[i+1+(k+2)*lda] * B[k+2+j*lda] + A[i+1+(k+3)*lda] * B[k+3+j*lda];
        // cij20 += A[i+2+(k)*lda] * B[k+j*lda] + A[i+2+(k+1)*lda] * B[k+1+j*lda] + A[i+2+(k+2)*lda] * B[k+2+j*lda] + A[i+2+(k+3)*lda] * B[k+3+j*lda];
        // cij30 += A[i+3+(k)*lda] * B[k+j*lda] + A[i+3+(k+1)*lda] * B[k+1+j*lda] + A[i+3+(k+2)*lda] * B[k+2+j*lda] + A[i+3+(k+3)*lda] * B[k+3+j*lda];
        
        // cij01 += A[i+k*lda] * B[k+(j+1)*lda] + A[i+(k+1)*lda] * B[k+1+(j+1)*lda] + A[i+(k+2)*lda] * B[k+2+(j+1)*lda] + A[i+(k+3)*lda] * B[k+3+(j+1)*lda];
        // cij11 += A[i+1+(k)*lda] * B[k+(j+1)*lda] + A[i+1+(k+1)*lda] * B[k+1+(j+1)*lda] + A[i+1+(k+2)*lda] * B[k+2+(j+1)*lda] + A[i+1+(k+3)*lda] * B[k+3+(j+1)*lda];
        // cij21 += A[i+2+(k)*lda] * B[k+(j+1)*lda] + A[i+2+(k+1)*lda] * B[k+1+(j+1)*lda] + A[i+2+(k+2)*lda] * B[k+2+(j+1)*lda] + A[i+2+(k+3)*lda] * B[k+3+(j+1)*lda];
        // cij31 += A[i+3+(k)*lda] * B[k+(j+1)*lda] + A[i+3+(k+1)*lda] * B[k+1+(j+1)*lda] + A[i+3+(k+2)*lda] * B[k+2+(j+1)*lda] + A[i+3+(k+3)*lda] * B[k+3+(j+1)*lda];
        
        // cij02 += A[i+k*lda] * B[k+(j+2)*lda] + A[i+(k+1)*lda] * B[k+1+(j+2)*lda] + A[i+(k+2)*lda] * B[k+2+(j+2)*lda] + A[i+(k+3)*lda] * B[k+3+(j+2)*lda];
        // cij12 += A[i+1+(k)*lda] * B[k+(j+2)*lda] + A[i+1+(k+1)*lda] * B[k+1+(j+2)*lda] + A[i+1+(k+2)*lda] * B[k+2+(j+2)*lda] + A[i+1+(k+3)*lda] * B[k+3+(j+2)*lda];
        // cij22 += A[i+2+(k)*lda] * B[k+(j+2)*lda] + A[i+2+(k+1)*lda] * B[k+1+(j+2)*lda] + A[i+2+(k+2)*lda] * B[k+2+(j+2)*lda] + A[i+2+(k+3)*lda] * B[k+3+(j+2)*lda];
        // cij32 += A[i+3+(k)*lda] * B[k+(j+2)*lda] + A[i+3+(k+1)*lda] * B[k+1+(j+2)*lda] + A[i+3+(k+2)*lda] * B[k+2+(j+2)*lda] + A[i+3+(k+3)*lda] * B[k+3+(j+2)*lda];
        
        // cij03 += A[i+k*lda] * B[k+(j+3)*lda] + A[i+(k+1)*lda] * B[k+1+(j+3)*lda] + A[i+(k+2)*lda] * B[k+2+(j+3)*lda] + A[i+(k+3)*lda] * B[k+3+(j+3)*lda];
        // cij13 += A[i+1+(k)*lda] * B[k+(j+3)*lda] + A[i+1+(k+1)*lda] * B[k+1+(j+3)*lda] + A[i+1+(k+2)*lda] * B[k+2+(j+3)*lda] + A[i+1+(k+3)*lda] * B[k+3+(j+3)*lda];
        // cij23 += A[i+2+(k)*lda] * B[k+(j+3)*lda] + A[i+2+(k+1)*lda] * B[k+1+(j+3)*lda] + A[i+2+(k+2)*lda] * B[k+2+(j+3)*lda] + A[i+2+(k+3)*lda] * B[k+3+(j+3)*lda];
        // cij33 += A[i+3+(k)*lda] * B[k+(j+3)*lda] + A[i+3+(k+1)*lda] * B[k+1+(j+3)*lda] + A[i+3+(k+2)*lda] * B[k+2+(j+3)*lda] + A[i+3+(k+3)*lda] * B[k+3+(j+3)*lda];
      }
      // C[i+j*lda] = cij00;
      // C[i+1+j*lda] = cij10;
      // C[i+2+j*lda] = cij20;
      // C[i+3+j*lda] = cij30;

      // C[i+(j+1)*lda] = cij01;
      // C[i+1+(j+1)*lda] = cij11;
      // C[i+2+(j+1)*lda] = cij21;
      // C[i+3+(j+1)*lda] = cij31;

      // C[i+(j+2)*lda] = cij02;
      // C[i+1+(j+2)*lda] = cij12;
      // C[i+2+(j+2)*lda] = cij22;
      // C[i+3+(j+2)*lda] = cij32;

      // C[i+(j+3)*lda] = cij03;
      // C[i+1+(j+3)*lda] = cij13;
      // C[i+2+(j+3)*lda] = cij23;
      // C[i+3+(j+3)*lda] = cij33;

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


// static void smallblock_dgemm (int lda, int M, int N, int K, double* A, double* B, double* C)
// {
//   /* Accumulate block dgemms into block of C */
//   for (int k = 0; k < K; k += K_BLOCK)
//   {
//     /* For each block-row of A */
//     for (int i = 0; i < M; i += I_BLOCK)
//     {
// //      memcpy(Ablock, A + i + k*lda, sizeof(double)*SMALL_BLOCK_1*SMALL_BLOCK_1);
//       /* For each block-column of B */
//       for (int j = 0; j < N; j += J_BLOCK)
//       {
//           /* Correct block dimensions if block "goes off edge of" the matrix */
//           int M1 = min (SMALL_BLOCK_1, M-i);
//           int N1 = min (SMALL_BLOCK_1, N-j);
//           int K1 = min (SMALL_BLOCK_1, K-k);

//           /* Perform individual block dgemm */
//           do_block(lda, M1, N1, K1, A+i+k*lda, B + k + j*lda, C + i + j*lda);
//       }
//     }
//   }
// }

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


  double* restrict Aalign;
  double* restrict Balign;
  double* restrict Calign;
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
  for (int j = 0; j < newsize; j += J_BLOCK)
  {
    /* Accumulate block dgemms into block of C */
    for (int i = 0; i < newsize; i += I_BLOCK)
    {
      /* For each block-row of A */ 
      for (int k = 0; k < newsize; k += K_BLOCK)
      {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int M = min (I_BLOCK, newsize-i);
          int N = min (J_BLOCK, newsize-j);
          int K = min (K_BLOCK, newsize-k);
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



