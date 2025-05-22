#ifndef CORRELATION_H
# define CORRELATION_H

/* Default to STANDARD_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define STANDARD_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(N) && !defined(M)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define N 32
#   define M 32
#  endif

#  ifdef SMALL_DATASET
#   define N 500
#   define M 500
#  endif

#  ifdef STANDARD_DATASET /* Default if unspecified. */
#   define N 1000
#   define M 1000
#  endif

#  ifdef LARGE_DATASET
#   define N 2000
#   define M 2000
#  endif

#  ifdef EXTRALARGE_DATASET
#   define N 4000
#   define M 4000
#  endif
# endif /* !N */

# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

/* Define macro to get execution time for a single function. */
#ifdef BENCHMARK
  #define START_TIMER cudaEvent_t start, stop;  \
                      cudaEventCreate(&start);  \
                      cudaEventCreate(&stop);   \
                      cudaEventRecord(start);
  #define STOP_TIMER  cudaEventRecord(stop);        \
                      cudaEventSynchronize(stop);       \
                      float milliseconds = 0;         \
                      cudaEventElapsedTime(&milliseconds, start, stop); \
                      printf("Execution time: %f ms\n", milliseconds);
#else
  #define START_TIMER
  #define STOP_TIMER
#endif

#endif /* !CORRELATION_H */
