#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <openacc.h>

using namespace std;

void display_grid(const float* data, int n, const char* header) {
    cout << header << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << setw(8) << fixed << setprecision(4) << data[i * n + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int N = 4;
    int num_elements = N * N;
    vector<float> A(num_elements);
    
    float input_values[] = {
        6.0f, 2.0f, 1.0f, 0.5f,
        2.0f, 7.0f, 3.0f, 1.0f,
        1.0f, 3.0f, 9.0f, 2.0f,
        0.5f, 1.0f, 2.0f, 8.0f
    };

    for (int i = 0; i < num_elements; ++i) A[i] = input_values[i];

    float *A_ptr = A.data();

    display_grid(A_ptr, N, "--- Input Matrix ---");

    cout << "Starting LU Decomposition (N=" << N << ")..." << endl;

    auto start_time = chrono::high_resolution_clock::now();

    #pragma acc data copy(A_ptr[0:num_elements])
    {
        for (int k = 0; k < N - 1; ++k) {
            
            #pragma acc parallel loop present(A_ptr)
            for (int i = k + 1; i < N; ++i) {
                A_ptr[i * N + k] /= A_ptr[k * N + k];
            }

            #pragma acc parallel loop collapse(2) present(A_ptr)
            for (int i = k + 1; i < N; ++i) {
                for (int j = k + 1; j < N; ++j) {
                    A_ptr[i * N + j] -= A_ptr[i * N + k] * A_ptr[k * N + j];
                }
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end_time - start_time;
    cout << "Time Elapsed: " << elapsed.count() << " ms" << endl;

    cout << "\nLower Triangular (L):" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float v = 0.0f;
            if (i == j) v = 1.0f;
            else if (i > j) v = A[i * N + j];
            cout << setw(8) << fixed << setprecision(4) << v << " ";
        }
        cout << endl;
    }

    cout << "\nUpper Triangular (U):" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float v = 0.0f;
            if (i <= j) v = A[i * N + j];
            cout << setw(8) << fixed << setprecision(4) << v << " ";
        }
        cout << endl;
    }

    vector<float> L(num_elements, 0.0f);
    vector<float> U(num_elements, 0.0f);
    
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            if(i == j) L[i * N + j] = 1.0f;
            else if(i > j) L[i * N + j] = A[i * N + j];
            
            if(i <= j) U[i * N + j] = A[i * N + j];
        }
    }
    
    float max_diff = 0.0f;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for(int k = 0; k < N; ++k) {
                sum += L[i * N + k] * U[k * N + j];
            }
            float diff = abs(sum - input_values[i * N + j]);
            if(diff > max_diff) max_diff = diff;
        }
    }
    cout << "\nValidation Error: " << max_diff << endl;

    return 0;
}
