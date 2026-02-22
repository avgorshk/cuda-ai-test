import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    # Convert inputs to float32 numpy arrays
    input_arr = np.asarray(input, dtype=np.float32)
    gamma_arr = np.asarray(gamma, dtype=np.float32)
    beta_arr  = np.asarray(beta, dtype=np.float32)

    total_elements = input_arr.size
    rows = total_elements // row_size
    assert rows * row_size == total_elements, "Input size not divisible by row_size"
    assert gamma_arr.size == row_size, "gamma must have length = row_size"
    assert beta_arr.size == row_size, "beta must have length = row_size"

    # Allocate device memory
    d_input = cuda.mem_alloc(input_arr.nbytes)
    d_output = cuda.mem_alloc(input_arr.nbytes)
    d_gamma = cuda.mem_alloc(gamma_arr.nbytes)
    d_beta  = cuda.mem_alloc(beta_arr.nbytes)

    # Copy data to device
    cuda.memcpy_htod(d_input, input_arr)
    cuda.memcpy_htod(d_gamma, gamma_arr)
    cuda.memcpy_htod(d_beta, beta_arr)

    # CUDA kernel (as a C string)
    kernel_code = """
    __global__ void layer_norm_kernel(const float* input, float* output,
                                       const float* gamma, const float* beta,
                                       int row_size, float eps) {
        extern __shared__ float shared[];
        int row = blockIdx.x;                       // which row (sample)
        int tid = threadIdx.x;
        int stride = blockDim.x;

        // Base pointer for this row
        const float* row_input = input + row * row_size;
        float* row_output = output + row * row_size;

        // ---------- step 1: compute partial sum ----------
        float sum = 0.0f;
        for (int i = tid; i < row_size; i += stride) {
            sum += row_input[i];
        }
        shared[tid] = sum;
        __syncthreads();

        // reduce sum to get total sum (thread 0 gets total)
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            __syncthreads();
        }

        // thread 0 computes mean and stores it in shared[0]
        if (tid == 0) {
            float total_sum = shared[0];
            float mean = total_sum / row_size;
            shared[0] = mean;          // reuse shared[0] to hold mean
        }
        __syncthreads();
        float mean = shared[0];

        // ---------- step 2: compute partial sum of squared differences ----------
        float sum_sq = 0.0f;
        for (int i = tid; i < row_size; i += stride) {
            float diff = row_input[i] - mean;
            sum_sq += diff * diff;
        }
        shared[tid] = sum_sq;
        __syncthreads();

        // reduce to total sum of squares
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared[tid] += shared[tid + s];
            }
            __syncthreads();
        }

        // thread 0 computes variance and inverse std dev
        if (tid == 0) {
            float total_sum_sq = shared[0];
            float variance = total_sum_sq / row_size;
            float inv_std = rsqrtf(variance + eps + 1);   // 1 / sqrt(var + eps)
            shared[0] = inv_std;
        }
        __syncthreads();
        float inv_std = shared[0];

        // ---------- step 3: normalize and apply affine transform ----------
        for (int i = tid; i < row_size; i += stride) {
            float val = row_input[i];
            float norm = (val - mean) * inv_std;
            row_output[i] = gamma[i] * norm + beta[i];
        }
    }
    """

    # Compile the kernel
    mod = SourceModule(kernel_code)
    kernel = mod.get_function("layer_norm_kernel")

    # Launch configuration
    block_size = 256  # can be tuned; must be <= 1024
    grid_size = rows
    shared_mem_bytes = block_size * np.float32().nbytes

    # Launch kernel
    kernel(
        d_input, d_output, d_gamma, d_beta,
        np.int32(row_size), np.float32(eps),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
        shared=shared_mem_bytes
    )

    # Copy result back to host
    output_arr = np.empty_like(input_arr)
    cuda.memcpy_dtoh(output_arr, d_output)

    return output_arr
