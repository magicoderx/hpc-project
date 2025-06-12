#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
// Add OpenMP library
#include <omp.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1000. */
#include "correlation.h"

/* Array initialization. */
static void init_array (int m, int n, DATA_TYPE *float_n, DATA_TYPE POLYBENCH_2D(data,M,N,m,n)){
  int i, j;

  *float_n = 1.2;

  for (i = 0; i < m; i++){
    for (j = 0; j < n; j++){
      data[i][j] = ((DATA_TYPE) i*j) / M;
    }
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int m, DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m)){
  int i, j;

  for (i = 0; i < m; i++){
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_correlation(int m, int n, DATA_TYPE float_n,	DATA_TYPE POLYBENCH_2D(data,M,N,m,n),	DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m),	DATA_TYPE POLYBENCH_1D(mean,M,m),	DATA_TYPE POLYBENCH_1D(stddev,M,m)){
  int i, j, j1, j2;

  DATA_TYPE eps = 0.1f;

#define sqrt_of_array_cell(x,j) sqrt(x[j])

  /* 1. Determine mean of column vectors of input data matrix */
  START_TIMER
  // Transfer all data to the GPU (and vice versa)
  #pragma omp target data map(tofrom: data[0:_PB_M][0:_PB_N], symmat[0:_PB_M][0:_PB_M], mean[0:_PB_M], stddev[0:_PB_M])
  {
    #pragma omp target teams distribute parallel for private(i) // Execute on GPU, distributing on cluster of threads. Jobs are divided between team's threads (loops are parallelized)
    for (j = 0; j < _PB_M; j++){
      mean[j] = 0.0;
	    for (i = 0; i < _PB_N; i++){
	        mean[j] += data[i][j];
      }
	    mean[j] /= float_n;
    }
  STOP_TIMER
    
  /* 2. Determine standard deviations of column vectors of data matrix. */
  START_TIMER
  #pragma omp target teams distribute parallel for private(i)
  for (j = 0; j < _PB_M; j++){
    stddev[j] = 0.0;
	  for (i = 0; i < _PB_N; i++){
	    stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    }
	  stddev[j] /= float_n;
	  stddev[j] = sqrt_of_array_cell(stddev, j);
	/* The following in an inelegant but usual way to handle
	   near-zero std. dev. values, which below would cause a zero-
	   divide. */
	  stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
  }
  STOP_TIMER
    
  /* 3. Center and reduce the column vectors. */
  START_TIMER
  #pragma omp target teams distribute parallel for collapse(2) // Collaps the loops in order to avoid cration of thread for each i (this would have meant execute all j iterations in sequence)
  for (i = 0; i < _PB_N; i++){
    for (j = 0; j < _PB_M; j++){
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }
  }
  STOP_TIMER
    
  /* 4. Calculate the m * m correlation matrix. */
  START_TIMER
  #pragma omp target teams distribute parallel for private(j2, i)
  for (j1 = 0; j1 < _PB_M-1; j1++){
    symmat[j1][j1] = 1.0; // Set diagonal values to 1.0
	  for (j2 = j1+1; j2 < _PB_M; j2++){
      DATA_TYPE sum = 0.0; // Initialize new variable to avoid conflicts
	    for (i = 0; i < _PB_N; i++){
        sum += data[i][j1] * data[i][j2];
      }
      symmat[j1][j2] = sum;
      symmat[j2][j1] = sum;
    }
  }
}
  symmat[_PB_M-1][_PB_M-1] = 1.0;
  STOP_TIMER
}

int main(int argc, char** argv){
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  DATA_TYPE float_n;
  POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,M,m,m);
  POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);

  /* Initialize array(s). */
  init_array (m, n, &float_n, POLYBENCH_ARRAY(data));
  
  /* Start timer. */
  polybench_start_instruments;
  /* Run kernel. */
  kernel_correlation (m, n, float_n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev));
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;
  
  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(data);
  POLYBENCH_FREE_ARRAY(symmat);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(stddev);

  return 0;
}
