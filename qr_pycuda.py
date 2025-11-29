import numpy as np
import pycuda.driver as cuda_driver
import pycuda.autoinit
from pycuda.compiler import SourceModule

cuda_kernels = """
__global__ void compute_squared_norm(float *data, int n, int col_idx, float *norm_sq) {
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    
    for (int i = tid; i < n; i += blockDim.x) {
        float val = data[i * n + col_idx];
        sum_sq += val * val;
    }
    
    __shared__ float s_mem[256];
    s_mem[tid] = sum_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) *norm_sq = s_mem[0];
}

__global__ void normalize_vector(float *data, int n, int col_idx, float *norm_sq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float norm = sqrtf(*norm_sq);
    
    if (idx < n) {
        data[idx * n + col_idx] /= norm;
    }
}

__global__ void calculate_dots(float *data, int n, int pivot_idx, float *dots_out) {
    int current_col = pivot_idx + 1 + blockIdx.x;
    int tid = threadIdx.x;
    
    if (current_col < n) {
        float dot_val = 0.0f;
        for (int i = tid; i < n; i += blockDim.x) {
            dot_val += data[i * n + pivot_idx] * data[i * n + current_col];
        }
        
        __shared__ float s_mem[256];
        s_mem[tid] = dot_val;
        __syncthreads();
        
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_mem[tid] += s_mem[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) dots_out[current_col] = s_mem[0];
    }
}

__global__ void update_matrix(float *data, int n, int pivot_idx, float *dots_in) {
    int current_col = pivot_idx + 1 + blockIdx.x;
    int tid = threadIdx.x;
    
    if (current_col < n) {
        float coeff = dots_in[current_col];
        for (int i = tid; i < n; i += blockDim.x) {
            data[i * n + current_col] -= coeff * data[i * n + pivot_idx];
        }
    }
}
"""

class GPU_QR_Decomposition:
    def __init__(self):
        self.module = SourceModule(cuda_kernels)
        self.kernel_sq_norm = self.module.get_function("compute_squared_norm")
        self.kernel_norm_col = self.module.get_function("normalize_vector")
        self.kernel_dots = self.module.get_function("calculate_dots")
        self.kernel_update = self.module.get_function("update_matrix")

    def execute(self, host_matrix):
        rows = host_matrix.shape[0]
        
        device_matrix = cuda_driver.mem_alloc(host_matrix.nbytes)
        cuda_driver.memcpy_htod(device_matrix, host_matrix)
        
        device_scalar = cuda_driver.mem_alloc(4)
        device_row_buffer = cuda_driver.mem_alloc(rows * 4)
        
        host_R = np.zeros((rows, rows), dtype=np.float32)
        
        start_event = cuda_driver.Event()
        end_event = cuda_driver.Event()
        
        start_event.record()
        
        for k in range(rows):
            self.kernel_sq_norm(device_matrix, np.int32(rows), np.int32(k), device_scalar, block=(256, 1, 1), grid=(1, 1))
            
            scalar_buffer = np.zeros(1, dtype=np.float32)
            cuda_driver.memcpy_dtoh(scalar_buffer, device_scalar)
            host_R[k, k] = np.sqrt(scalar_buffer[0])
            
            grid_dim_norm = ((rows + 255) // 256, 1)
            self.kernel_norm_col(device_matrix, np.int32(rows), np.int32(k), device_scalar, block=(256, 1, 1), grid=grid_dim_norm)
            
            if k < rows - 1:
                remaining = rows - 1 - k
                self.kernel_dots(device_matrix, np.int32(rows), np.int32(k), device_row_buffer, block=(256, 1, 1), grid=(remaining, 1))
                
                temp_row = np.zeros(rows, dtype=np.float32)
                cuda_driver.memcpy_dtoh(temp_row, device_row_buffer)
                host_R[k, k+1:] = temp_row[k+1:]
                
                self.kernel_update(device_matrix, np.int32(rows), np.int32(k), device_row_buffer, block=(256, 1, 1), grid=(remaining, 1))
                
        end_event.record()
        end_event.synchronize()
        print(f"GPU Time: {start_event.time_till(end_event):.4f} ms")
        
        host_Q = np.empty_like(host_matrix)
        cuda_driver.memcpy_dtoh(host_Q, device_matrix)
        
        return host_Q, host_R

if __name__ == "__main__":
    N = 4
    print(f"Dimensions: {N}x{N}")
    
    input_data = np.array([
        [2.0, 0.5, 1.0, 0.0],
        [0.5, 2.0, 1.0, 0.0],
        [1.0, 1.0, 2.0, 0.5],
        [0.0, 0.5, 0.5, 2.0]
    ], dtype=np.float32)
    
    matrix_A = input_data.copy()
    
    print("Source Matrix:")
    for i in range(N):
        row_str = " ".join([f"{val:8.4f}" for val in matrix_A[i]])
        print(row_str)
        
    qr_solver = GPU_QR_Decomposition()
    Q_mat, R_mat = qr_solver.execute(matrix_A)
    
    print("\nMatrix Q:")
    for i in range(N):
        row_str = " ".join([f"{val:8.4f}" for val in Q_mat[i]])
        print(row_str)
        
    print("\nMatrix R:")
    for i in range(N):
        row_str = " ".join([f"{val:8.4f}" for val in R_mat[i]])
        print(row_str)
        
    reconstructed = np.dot(Q_mat, R_mat)
    max_error = np.max(np.abs(input_data - reconstructed))
    print(f"\nError Check: {max_error:.6f}")
