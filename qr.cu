#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>

using namespace std;

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "GPU Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << endl; \
            exit(EXIT_FAILURE); \
        } \
    }

__global__ void k_calc_sq_norm(float *matrix, int rows, int col_idx, float *d_res) {
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    
    for (int i = tid; i < rows; i += blockDim.x) {
        float val = matrix[i * rows + col_idx];
        sum_sq += val * val;
    }
    
    __shared__ float s_red[256];
    s_red[tid] = sum_sq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_red[tid] += s_red[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) *d_res = s_red[0];
}

__global__ void k_normalize_col(float *matrix, int rows, int col_idx, float *d_norm_sq) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float norm = sqrtf(*d_norm_sq);
    
    if (gid < rows) {
        matrix[gid * rows + col_idx] /= norm;
    }
}

__global__ void k_dot_product(float *matrix, int rows, int pivot_col, float *d_dots) {
    int target_col = pivot_col + 1 + blockIdx.x;
    int tid = threadIdx.x;
    
    if (target_col < rows) {
        float dot = 0.0f;
        for (int i = tid; i < rows; i += blockDim.x) {
            dot += matrix[i * rows + pivot_col] * matrix[i * rows + target_col];
        }
        
        __shared__ float s_red[256];
        s_red[tid] = dot;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_red[tid] += s_red[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) d_dots[target_col] = s_red[0];
    }
}

__global__ void k_update_cols(float *matrix, int rows, int pivot_col, float *d_dots) {
    int target_col = pivot_col + 1 + blockIdx.x;
    int tid = threadIdx.x;
    
    if (target_col < rows) {
        float coeff = d_dots[target_col];
        for (int i = tid; i < rows; i += blockDim.x) {
            matrix[i * rows + target_col] -= coeff * matrix[i * rows + pivot_col];
        }
    }
}

void perform_qr_decomposition(float *d_mat, int n, float *h_R) {
    float *d_sq_val, *d_projections;
    CHECK_CUDA(cudaMalloc(&d_sq_val, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_projections, n * sizeof(float)));

    for (int i = 0; i < n; ++i) {
        k_calc_sq_norm<<<1, 256>>>(d_mat, n, i, d_sq_val);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        float h_sq_val;
        CHECK_CUDA(cudaMemcpy(&h_sq_val, d_sq_val, sizeof(float), cudaMemcpyDeviceToHost));
        h_R[i * n + i] = sqrtf(h_sq_val);
        
        int num_blocks = (n + 255) / 256;
        k_normalize_col<<<num_blocks, 256>>>(d_mat, n, i, d_sq_val);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        if (i < n - 1) {
            int remaining_cols = n - 1 - i;
            k_dot_product<<<remaining_cols, 256>>>(d_mat, n, i, d_projections);
            CHECK_CUDA(cudaDeviceSynchronize());
            
            vector<float> h_projections(n);
            CHECK_CUDA(cudaMemcpy(h_projections.data(), d_projections, n * sizeof(float), cudaMemcpyDeviceToHost));
            
            for (int j = i + 1; j < n; ++j) {
                h_R[i * n + j] = h_projections[j];
            }
            
            k_update_cols<<<remaining_cols, 256>>>(d_mat, n, i, d_projections);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
    
    cudaFree(d_sq_val);
    cudaFree(d_projections);
}

int main() {
    int N = 4;
    size_t mat_size = N * N * sizeof(float);
    
    vector<float> h_A(N * N);
    vector<float> h_A_copy(N * N);
    vector<float> h_R(N * N, 0.0f);
    
    float init_data[] = {
        2.0f, 0.5f, 1.0f, 0.0f,
        0.5f, 2.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 2.0f, 0.5f,
        0.0f, 0.5f, 0.5f, 2.0f
    };

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = init_data[i];
        h_A_copy[i] = init_data[i];
    }

    cout << "--- Initial Matrix (A) ---" << endl;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cout << setw(8) << fixed << setprecision(4) << h_A[r * N + c] << " ";
        }
        cout << endl;
    }

    float *d_mat;
    CHECK_CUDA(cudaMalloc(&d_mat, mat_size));
    CHECK_CUDA(cudaMemcpy(d_mat, h_A.data(), mat_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    cout << "Starting QR Decomposition on GPU..." << endl;

    CHECK_CUDA(cudaEventRecord(start));
    perform_qr_decomposition(d_mat, N, h_R.data());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float millis = 0;
    CHECK_CUDA(cudaEventElapsedTime(&millis, start, stop));
    cout << "Time Elapsed: " << millis << " ms" << endl;

    CHECK_CUDA(cudaMemcpy(h_A.data(), d_mat, mat_size, cudaMemcpyDeviceToHost));

    cout << "--- Matrix Q ---" << endl;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cout << setw(8) << fixed << setprecision(4) << h_A[r * N + c] << " ";
        }
        cout << endl;
    }

    cout << "--- Matrix R ---" << endl;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cout << setw(8) << fixed << setprecision(4) << h_R[r * N + c] << " ";
        }
        cout << endl;
    }

    float max_error = 0.0f;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            float recon_val = 0.0f;
            for (int k = 0; k < N; ++k) {
                recon_val += h_A[r * N + k] * h_R[k * N + c];
            }
            float diff = abs(recon_val - h_A_copy[r * N + c]);
            if (diff > max_error) max_error = diff;
        }
    }
    cout << "\nReconstruction Error (|A - Q*R|): " << max_error << endl;

    cudaFree(d_mat);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
