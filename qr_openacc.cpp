#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <openacc.h>

using namespace std;

int main() {
    const int N = 4;
    vector<float> Q_matrix(N * N);
    vector<float> input_copy(N * N);
    
    float init_array[] = {
        2.0f, 0.5f, 1.0f, 0.0f,
        0.5f, 2.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 2.0f, 0.5f,
        0.0f, 0.5f, 0.5f, 2.0f
    };

    for (int i = 0; i < N * N; ++i) {
        Q_matrix[i] = init_array[i];
        input_copy[i] = init_array[i];
    }

    float *dev_Q = Q_matrix.data();
    vector<float> R_matrix(N * N, 0.0f);
    float *dev_R = R_matrix.data();

    cout << "--- Source Matrix ---" << endl;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cout << setw(8) << fixed << setprecision(4) << Q_matrix[r * N + c] << " ";
        }
        cout << endl;
    }

    cout << "Executing OpenACC QR Decomposition..." << endl;

    auto time_start = chrono::high_resolution_clock::now();

    #pragma acc data copy(dev_Q[0:N*N]) copyout(dev_R[0:N*N])
    {
        for (int k = 0; k < N; ++k) {
            float sum_sq = 0.0f;
            
            #pragma acc parallel loop reduction(+:sum_sq) present(dev_Q)
            for (int r = 0; r < N; ++r) {
                float val = dev_Q[r * N + k];
                sum_sq += val * val;
            }
            
            float norm = sqrt(sum_sq);
            
            #pragma acc serial present(dev_R)
            {
                dev_R[k * N + k] = norm;
            }
            
            #pragma acc parallel loop present(dev_Q)
            for (int r = 0; r < N; ++r) {
                dev_Q[r * N + k] /= norm;
            }
            
            #pragma acc parallel loop present(dev_Q, dev_R)
            for (int j = k + 1; j < N; ++j) {
                float dot_prod = 0.0f;
                
                #pragma acc loop reduction(+:dot_prod)
                for (int r = 0; r < N; ++r) {
                    dot_prod += dev_Q[r * N + k] * dev_Q[r * N + j];
                }
                
                dev_R[k * N + j] = dot_prod;
                
                #pragma acc loop
                for (int r = 0; r < N; ++r) {
                    dev_Q[r * N + j] -= dot_prod * dev_Q[r * N + k];
                }
            }
        }
    }

    auto time_end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = time_end - time_start;
    cout << "Compute Duration: " << duration.count() << " ms" << endl;

    cout << "--- Q Matrix ---" << endl;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cout << setw(8) << fixed << setprecision(4) << Q_matrix[r * N + c] << " ";
        }
        cout << endl;
    }

    cout << "--- R Matrix ---" << endl;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            cout << setw(8) << fixed << setprecision(4) << R_matrix[r * N + c] << " ";
        }
        cout << endl;
    }

    float max_error = 0.0f;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += Q_matrix[r * N + k] * R_matrix[k * N + c];
            }
            float diff = abs(sum - input_copy[r * N + c]);
            if (diff > max_error) max_error = diff;
        }
    }
    cout << "\nVerification Error: " << max_error << endl;

    return 0;
}
