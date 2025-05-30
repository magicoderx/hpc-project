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

# define _PB_N POLYBENCH_LOOP_BOUND(N,n)
# define _PB_M POLYBENCH_LOOP_BOUND(M,m)

# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif

/* Define macro to get execution time for a single function. */
#ifdef POLYBENCH_FUNCTION
  #define START_TIMER polybench_start_instruments;
  #define STOP_TIMER  polybench_stop_instruments; \
                      polybench_print_instruments;
#else
  #define START_TIMER
  #define STOP_TIMER
#endif

#endif /* !CORRELATION_H */
