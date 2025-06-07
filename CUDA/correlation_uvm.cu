#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
// Add CUDA library
#include <cuda.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 1000. */
#include "correlation.h"

#define DATA_TYPE float

// GPU error checking macro
// https://gist.github.com/jefflarkin/5390993
#define cudaCheckErr(){                                          \
    cudaError_t e = cudaGetLastError();                         \
    if (e != cudaSuccess){                                     \
        fprintf(stderr, "Cuda failure: %s\n", cudaGetErrorString(e));  \
        exit(0);                                       \
    }                                                             \
}

/* Array initialization. */
// This operation is exwcuted host side
void init_array(int m, int n, DATA_TYPE *float_n, DATA_TYPE *data){
    int i, j;
    
    *float_n = 1.2;
    
    for (i = 0; i < m; i++){
        for (j = 0; j < n; j++){
            data[i * n + j] = ((DATA_TYPE)i * j) / M;
        }
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
void print_array(int m, DATA_TYPE *symmat){
    int i, j;

    for (i = 0; i < m; i++){
        for (j = 0; j < m; j++){
            fprintf(stderr, DATA_PRINTF_MODIFIER, symmat[i * m + j]);
            if ((i * m + j) % 20 == 0) fprintf(stderr, "\n");
        }
    }
    fprintf (stderr, "\n");
}

// CUDA kernel for mean
__global__ void mean_kernel(const DATA_TYPE* data, DATA_TYPE* mean, int m, int n, DATA_TYPE float_n){
    // Index of column for thread
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread block check, avoid thread execute invalid columns
    if (j < m){
        DATA_TYPE sum = 0.0;
        for (int i = 0; i < n; i++){
            // Index times number of elements on the row, then add the value of the column. To get each row value for own column
            sum += data[i * m + j];
        }
        mean[j] = sum / float_n;
    }
}

// CUDA kernel for stddev
__global__ void stddev_kernel(const DATA_TYPE* data, const DATA_TYPE* mean, DATA_TYPE* stddev, int m, int n, DATA_TYPE float_n, DATA_TYPE eps){
    // Index of column for thread
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread block check, avoid thread execute invalid columns
    if (j < m){
        DATA_TYPE sum = 0.0;
        for (int i = 0; i < n; i++){
            DATA_TYPE d = data[i * m + j] - mean[j];
            sum += d * d;
        }
        sum /= float_n;
        // Use sqrt function optimize for float type
        sum = sqrtf(sum);
        /* The following in an inelegant but usual way to handle
        near-zero std. dev. values, which below would cause a zero-
        divide. */
        stddev[j] = sum <= eps ? 1.0f : sum;
    }
}

// CUDA kernel for normalization
__global__ void normalize_kernel(DATA_TYPE* data, const DATA_TYPE* mean, const DATA_TYPE* stddev, int m, int n, DATA_TYPE float_n){
    // Operations on data here are independend, so evaluate position of row (i) and column (j) of values. In order to let every thread execute on single cell of the matrix
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread block check, avoid thread execute invalid columns and rows
    if (i < n && j < m){
        data[i * m + j] -= mean[j];
        data[i * m + j] /= sqrtf(float_n) * stddev[j];
    }
}

// CUDA kernel for correlation matrix
__global__ void correlation_kernel(const DATA_TYPE* data, DATA_TYPE* symmat, int m, int n){
    // Operations on data here are indepentend, so evaluate position of row (i) and column (j) of values. In order to let every thread execute on single cell of the matrix
    int j1 = blockIdx.y * blockDim.y + threadIdx.y;
    int j2 = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread block check, avoid thread execute invalid columns and rows
    if (j1 < m && j2 < m){
        // Diagonal values are set to 1.0
        if (j1 == j2){
            symmat[j1 * m + j2] = 1.0;
        // Compute only on opper side of the matix (bottom side is symmetric)
        } else if (j2 > j1){
            DATA_TYPE sum = 0.0;
            for (int i = 0; i < n; i++){
                sum += data[i * m + j1] * data[i * m + j2];
            }
            symmat[j1 * m + j2] = sum;
            symmat[j2 * m + j1] = sum; // Symmetric
        }
    }
}

int main(){
    /* Retrieve problem size. */
    int n = N;
    int m = M;

    /* Variable declaration/allocation. */
    DATA_TYPE float_n;
    DATA_TYPE *h_data, *h_mean, *h_stddev, *h_symmat;

    // Memory allocation host side (UVM solution)
    cudaMallocManaged(&h_data,   m * n * sizeof(DATA_TYPE));
    cudaMallocManaged(&h_mean,   m * sizeof(DATA_TYPE));
    cudaMallocManaged(&h_stddev, m * sizeof(DATA_TYPE));
    cudaMallocManaged(&h_symmat, m * m * sizeof(DATA_TYPE));

    /* Initialize array(s). */
    init_array(m, n, &float_n, h_data);

    /* =============================================================== */
    /* === No allocation on device since use virtual shared memory === */
    /* =============================================================== */

    // CUDA cannot use polybench, create cudaEvent to get execution time information (defined in correlation.h)
    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    START_TIMER

    /* 1. Determine mean of column vectors of input data matrix */
    // Chosen 256 threads per block for support on most of GPUs 
    int blocksize = 256;
    int num_blocks = (m + blocksize - 1) / blocksize;
    mean_kernel<<<num_blocks, blocksize>>>(h_data, h_mean, m, n, float_n);
    cudaCheckErr();

    /* 2. Determine standard deviations of column vectors of data matrix. */
    stddev_kernel<<<num_blocks, blocksize>>>(h_data, h_mean, h_stddev, m, n, float_n, 0.1f);
    cudaCheckErr();

    /* 3. Center and reduce the column vectors. */
    // Create a bidimensional block (16x16) to let the thread operate on a single cell of the matrix
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#dim3
    int bsize2d = 16;
    dim3 blocksize2d(bsize2d, bsize2d);
    dim3 num_blocks2d((m + bsize2d - 1) / 16, (n + bsize2d - 1) / 16);
    normalize_kernel<<<num_blocks2d, blocksize2d>>>(h_data, h_mean, h_stddev, m, n, float_n);
    cudaCheckErr();

    /* 4. Calculate the m * m correlation matrix. */
    correlation_kernel<<<num_blocks2d, blocksize2d>>>(h_data, h_symmat, m, n);
    cudaCheckErr();

    // Synchronize device befor finish
    cudaDeviceSynchronize();

    #ifdef DUMP_ARRAYS
        print_array(m, h_symmat);
    #endif

    // Cleanup
    cudaFree(h_data);
    cudaFree(h_mean);
    cudaFree(h_stddev);
    cudaFree(h_symmat);

    // Stop the timer and print info
    STOP_TIMER

    return 0;
}
