#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024 * 10
#define M 1024 * 20
#define L 1024 * 30

float randomNumber()
{
    /***
     * Returns a random number in range [-2, 2)
     */
    return ((float)rand() / (float)RAND_MAX - 0.5f) * 4;
}

__global__ void matmul(float *src1, float *src2, float *dest, int n, int m, int l)
{
    /**
     * Kernel that does the matrix multiplication in "blocks" (i.e. each thread computes a small sub-matrix)
     * NOTE: This kernel does not work if the number of elements in the matrices are not entirely divisible by the number of threads
     */

    // X, Y start position of block
    const int block_begin_x = (n * blockIdx.x / gridDim.x);
    const int block_begin_y = (l * blockIdx.y / gridDim.y);

    // Number of elements to be computed by a single thread
    const int n_elements_x = (n / (gridDim.x * blockDim.x));
    const int n_elements_y = (l / (gridDim.y * blockDim.y));

    // X, Y absolute start position of thread
    const int thread_begin_x = block_begin_x + threadIdx.x * n_elements_x;
    const int thread_begin_y = block_begin_y + threadIdx.y * n_elements_y;

    // Computing partial result for selected block
    for (int x = thread_begin_x; x < thread_begin_x + n_elements_x; x++)
    {
        for (int y = thread_begin_y; y < thread_begin_y + n_elements_y; y++)
        {
            for (int i = 0; i < m; i++)
            {
                dest[x * l + y] += src1[x * m + i] * src2[i * l + y];
            }
        }
    }
}

void cpuMatMul(float *src1, float *src2, float *dest, int n, int m, int l)
{
    /***
     * CPU matrix multiplication. To be used to compare computation times.
    */
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < l; j++)
        {
            for (int k = 0; k < m; k++)
            {
                dest[i * l + j] += src1[i * m + k] * src2[k * l + j];
            }
        }
    }
}

int main()
{
    // Setting random
    srand((unsigned int)time(NULL));

    // Initializing matrices in CPU (host)
    float *a = (float *)malloc(sizeof(float) * N * M);
    float *b = (float *)malloc(sizeof(float) * M * L);
    float *c = (float *)malloc(sizeof(float) * N * L);

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[j * M + i] = randomNumber();
        }

        for (int j = 0; j < L; j++)
        {
            b[i * L + j] = randomNumber();
        }
    }

    for (int i = 0; i < N * L; i++)
    {
        c[i] = 0;
    }

    // Allocating GPU (device) memory
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, sizeof(float) * N * M);
    cudaMalloc((void **)&dev_b, sizeof(float) * M * L);
    cudaMalloc((void **)&dev_c, sizeof(float) * N * L);

    // Copying matrices to GPU (device)
    cudaMemcpy(dev_a, a, sizeof(float) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float) * M * L, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, sizeof(float) * N * L, cudaMemcpyHostToDevice);

    // Performing computation on GPU (device)
    // Notation: <<< # Blocks, # Threads per block >>>
    // There can't be more than 2**16 -1 = 65'535 blocks per dimension
    // There can't be more than 2**10 = 1024 threads per block
    // Total maximum threads: 2**26 = 67M
    dim3 n_blocks(4, 4, 1);
    dim3 n_threads(64, 64, 1);
    matmul<<<n_blocks, n_threads>>>(dev_a, dev_b, dev_c, N, M, L);

    // Copying result back to the CPU (host)
    cudaMemcpy(c, dev_c, sizeof(float) * N * L, cudaMemcpyDeviceToHost);

    return 0;
}