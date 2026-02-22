#include "softmax_cuda.h"

#include <assert.h>
#include <cuda.h>

#define BLOCK_SIZE 256

__device__ float reduce_max(float value, float* buffer) {
    buffer[threadIdx.x] = value;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float left = buffer[threadIdx.x];
            float right = buffer[threadIdx.x + s];
            buffer[threadIdx.x] = (left < right ? right : left);
        }
        __syncthreads();
    }

    return buffer[0];
}

__device__ float reduce_sum(float value, float* buffer) {
    buffer[threadIdx.x] = value;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            buffer[threadIdx.x] += buffer[threadIdx.x + s];
        }
        __syncthreads();
    }

    return buffer[0];
}

__global__ void Softmax(const float* input, float* output, int row_size, int col_size) {
    __shared__ float buffer[BLOCK_SIZE];

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float value = input[blockIdx.x * row_size + i];
        if (value > local_max) local_max = value;
    }

    float max = reduce_max(local_max, buffer);
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float value = input[blockIdx.x * row_size + i];
        local_sum += expf(value - max);
    }

    float sum = reduce_sum(local_sum, buffer);
    __syncthreads();

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float value = input[blockIdx.x * row_size + i];
        output[blockIdx.x * row_size + i] = expf(value - max) / sum;
    }
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_size) {
    cudaError_t status = cudaSuccess;

    float* dev_input = nullptr;
    status = cudaMalloc(&dev_input, input.size() * sizeof(float));
    assert(status == cudaSuccess);

    float* dev_output = nullptr;
    status = cudaMalloc(&dev_output, input.size() * sizeof(float));
    assert(status == cudaSuccess);

    status = cudaMemcpy(
        dev_input, input.data(), input.size() * sizeof(float),
        cudaMemcpyHostToDevice);
    assert(status == cudaSuccess);

    int col_size = input.size() / row_size;
    assert(row_size * col_size == input.size());

    Softmax<<<col_size, BLOCK_SIZE>>>(dev_input, dev_output, row_size, col_size);
    status = cudaDeviceSynchronize();
    assert(status == cudaSuccess);

    std::vector<float> output(input.size());
    status = cudaMemcpy(
        output.data(), dev_output,  input.size() * sizeof(float),
        cudaMemcpyDeviceToHost);
    assert(status == cudaSuccess);

    status = cudaFree(dev_input);
    assert(status == cudaSuccess);
    status = cudaFree(dev_output);
    assert(status == cudaSuccess);

    return output;
}
