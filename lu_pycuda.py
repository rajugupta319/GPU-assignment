import numpy as np
import pycuda.driver as cuda_drv
import pycuda.autoinit
from pycuda.compiler import SourceModule

cuda_code = """
__global__ void normalize_column(float *matrix, int dim, int pivot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = pivot + 1 + idx;
    
    if (row < dim) {
        int target_idx = row * dim + pivot;
        int pivot_idx = pivot * dim + pivot;
        matrix[target_idx] /= matrix[pivot_idx];
    }
}

__global__ void update_submatrix(float *matrix, int dim, int pivot) {
    int col_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int row_offset = blockIdx.y * blockDim.y + threadIdx.y;
    
    int col = pivot + 1 + col_offset;
    int row = pivot + 1 + row_offset;
    
    if (row < dim && col < dim) {
        int index = row * dim + col;
        int row_factor = row * dim + pivot;
        int col_factor = pivot * dim + col;
        
        matrix[index] -= matrix[row_factor] * matrix[col_factor];
    }
}
"""

class GPU_LU_Decomposition:
    def __init__(self):
        self.module = SourceModule(cuda_code)
        self.kernel_norm = self.module.get_function("normalize_column")
        self.kernel_update = self.module.get_function("update_submatrix")

    def execute(self, host_input):
        n = host_input.shape[0]
        device_ptr = cuda_drv.mem_alloc(host_input.nbytes)
        cuda_drv.memcpy_htod(device_ptr, host_input)
        
        start_evt = cuda_drv.Event()
        end_evt = cuda_drv.Event()
        
        start_evt.record()
        
        for k in range(n - 1):
            remaining = n - 1 - k
            
            threads_per_block_norm = (256, 1, 1)
            blocks_per_grid_norm = ((remaining + 255) // 256, 1)
            self.kernel_norm(device_ptr, np.int32(n), np.int32(k), block=threads_per_block_norm, grid=blocks_per_grid_norm)
            
            threads_per_block_update = (16, 16, 1)
            grid_dim = (remaining + 15) // 16
            blocks_per_grid_update = (grid_dim, grid_dim)
            self.kernel_update(device_ptr, np.int32(n), np.int32(k), block=threads_per_block_update, grid=blocks_per_grid_update)
            
        end_evt.record()
        end_evt.synchronize()
        
        elapsed_time = start_evt.time_till(end_evt)
        print(f"GPU Computation Time: {elapsed_time:.4f} ms")
        
        host_result = np.empty_like(host_input)
        cuda_drv.memcpy_dtoh(host_result, device_ptr)
        return host_result

if __name__ == "__main__":
    matrix_dim = 4
    print(f"Matrix Dimension: {matrix_dim}x{matrix_dim}")
    
    input_matrix = np.array([
        [6.0, 2.0, 1.0, 0.5],
        [2.0, 7.0, 3.0, 1.0],
        [1.0, 3.0, 9.0, 2.0],
        [0.5, 1.0, 2.0, 8.0]
    ], dtype=np.float32)
    
    working_matrix = input_matrix.copy()
    
    print("Input Matrix:")
    for i in range(matrix_dim):
        row_str = " ".join([f"{val:8.4f}" for val in working_matrix[i]])
        print(row_str)
    
    lu_solver = GPU_LU_Decomposition()
    combined_result = lu_solver.execute(working_matrix)
    
    lower_tri = np.tril(combined_result, -1) + np.eye(matrix_dim)
    upper_tri = np.triu(combined_result)

    print("\nLower Matrix (L):")
    for i in range(matrix_dim):
        row_str = " ".join([f"{val:8.4f}" for val in lower_tri[i]])
        print(row_str)

    print("\nUpper Matrix (U):")
    for i in range(matrix_dim):
        row_str = " ".join([f"{val:8.4f}" for val in upper_tri[i]])
        print(row_str)
        
    reconstructed = np.dot(lower_tri, upper_tri)
    max_error = np.max(np.abs(input_matrix - reconstructed))
    print(f"\nReconstruction Error: {max_error:.6f}")
