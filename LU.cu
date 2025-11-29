#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <cmath>

using namespace std;

inline void assert_gpu(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        cerr << "GPU Failure: " << cudaGetErrorString(code) << " | " << file << ":" << line << endl;
        exit(code);
    }
}
#define CHECK(ans) { assert_gpu((ans), __FILE__, __LINE__); }

__global__ void compute_column_multipliers(float* __restrict__ matrix, int dim, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int current_row = step + 1 + tid;
    
    if (current_row < dim) {
        int idx_target = current_row * dim + step;
        int idx_pivot = step * dim + step;
        matrix[idx_target] /= matrix[idx_pivot];
    }
}

__global__ void update_submatrix(float* __restrict__ matrix, int dim, int step) {
    int c = step + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    int r = step + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    
    if (r < dim && c < dim) {
        int idx_cell = r * dim + c;
        int idx_col = r * dim + step;
        int idx_row = step * dim + c;
        matrix[idx_cell] -= matrix[idx_col] * matrix[idx_row];
    }
}

void execute_lu_decomposition(float* dev_ptr, int n) {
    for (int k = 0; k < n - 1; ++k) {
        int remaining = n - 1 - k;
        
        int block_size = 256;
        int grid_size = (remaining + block_size - 1) / block_size;
        compute_column_multipliers<<<grid_size, block_size>>>(dev_ptr, n, k);
        cudaDeviceSynchronize();

        dim3 b_dim(16, 16);
        dim3 g_dim((remaining + 15) / 16, (remaining + 15) / 16);
        update_submatrix<<<g_dim, b_dim>>>(dev_ptr, n, k);
        cudaDeviceSynchronize();
    }
}

int main() {
    int n = 4;
    size_t mem_size = n * n * sizeof(float);
    
    vector<float> h_matrix(n * n);
    float init_data[] = {
        6.0f, 2.0f, 1.0f, 0.5f,
        2.0f, 7.0f, 3.0f, 1.0f,
        1.0f, 3.0f, 9.0f, 2.0f,
        0.5f, 1.0f, 2.0f, 8.0f
    };
    
    for (int i = 0; i < n * n; ++i) h_matrix[i] = init_data[i];

    cout << "--- Initial Matrix ---" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << setw(8) << fixed << setprecision(4) << h_matrix[i * n + j] << " ";
        }
        cout << endl;
    }

    float *d_matrix;
    CHECK(cudaMalloc(&d_matrix, mem_size));
    CHECK(cudaMemcpy(d_matrix, h_matrix.data(), mem_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    cout << "Running GPU LU Factorization (N=" << n << ")..." << endl;

    CHECK(cudaEventRecord(start));
    execute_lu_decomposition(d_matrix, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    cout << "Time Taken: " << elapsed_ms << " ms" << endl;

    CHECK(cudaMemcpy(h_matrix.data(), d_matrix, mem_size, cudaMemcpyDeviceToHost));
    
    vector<float> lower(n * n, 0.0f);
    vector<float> upper(n * n, 0.0f);

    cout << "\nLower Matrix (L):" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float val = 0.0f;
            if (i == j) val = 1.0f;
            else if (i > j) val = h_matrix[i * n + j];
            
            lower[i * n + j] = val;
            cout << setw(8) << fixed << setprecision(4) << val << " ";
        }
        cout << endl;
    }

    cout << "\nUpper Matrix (U):" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float val = 0.0f;
            if (i <= j) val = h_matrix[i * n + j];
            
            upper[i * n + j] = val;
            cout << setw(8) << fixed << setprecision(4) << val << " ";
        }
        cout << endl;
    }

    float max_error = 0.0f;
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < n; ++k) {
                sum += lower[i * n + k] * upper[k * n + j];
            }
            float diff = abs(sum - init_data[i * n + j]);
            if(diff > max_error) max_error = diff;
        }
    }
    cout << "\nReconstruction Error: " << max_error << endl;
    
    cudaFree(d_matrix);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
