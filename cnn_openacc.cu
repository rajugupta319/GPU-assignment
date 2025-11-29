#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <openacc.h>

using namespace std;

void display_matrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << mat[i * cols + j] << " ";
        }
        cout << endl;
    }
}

void convolution_forward(float* input_tensor, float* kernel, float* output_map, 
                         int batch_size, int in_channels, int out_channels, 
                         int h_in, int w_in, int k_dim, 
                         int h_out, int w_out) {
    
    int len_in = batch_size * in_channels * h_in * w_in;
    int len_kern = out_channels * in_channels * k_dim * k_dim;
    int len_out = batch_size * out_channels * h_out * w_out;

    #pragma acc data copyin(input_tensor[0:len_in], kernel[0:len_kern]) copyout(output_map[0:len_out])
    {
        #pragma acc parallel loop collapse(3)
        for (int b = 0; b < batch_size; ++b) {
            for (int f = 0; f < out_channels; ++f) {
                for (int r = 0; r < h_out; ++r) {
                    #pragma acc loop vector
                    for (int c = 0; c < w_out; ++c) {
                        float pixel_val = 0.0f;
                        for (int ch = 0; ch < in_channels; ++ch) {
                            for (int kr = 0; kr < k_dim; ++kr) {
                                for (int kc = 0; kc < k_dim; ++kc) {
                                    int in_row = r + kr;
                                    int in_col = c + kc;
                                    int src_idx = ((b * in_channels + ch) * h_in + in_row) * w_in + in_col;
                                    int w_idx = ((f * in_channels + ch) * k_dim + kr) * k_dim + kc;
                                    pixel_val += input_tensor[src_idx] * kernel[w_idx];
                                }
                            }
                        }
                        int dest_idx = ((b * out_channels + f) * h_out + r) * w_out + c;
                        output_map[dest_idx] = pixel_val;
                    }
                }
            }
        }
    }
}

void convolution_backward(float* input_tensor, float* grad_output, float* grad_kernel, 
                          int batch_size, int in_channels, int out_channels, 
                          int h_in, int w_in, int k_dim, 
                          int h_out, int w_out) {

    int len_in = batch_size * in_channels * h_in * w_in;
    int len_grad_out = batch_size * out_channels * h_out * w_out;
    int len_grad_w = out_channels * in_channels * k_dim * k_dim;

    #pragma acc data copyin(input_tensor[0:len_in], grad_output[0:len_grad_out]) copyout(grad_kernel[0:len_grad_w])
    {
        #pragma acc parallel loop collapse(2)
        for (int f = 0; f < out_channels; ++f) {
            for (int ch = 0; ch < in_channels; ++ch) {
                #pragma acc loop worker
                for (int kr = 0; kr < k_dim; ++kr) {
                    #pragma acc loop vector
                    for (int kc = 0; kc < k_dim; ++kc) {
                        float grad_acc = 0.0f;
                        for (int b = 0; b < batch_size; ++b) {
                            for (int r = 0; r < h_out; ++r) {
                                for (int c = 0; c < w_out; ++c) {
                                    int in_row = r + kr;
                                    int in_col = c + kc;
                                    int src_idx = ((b * in_channels + ch) * h_in + in_row) * w_in + in_col;
                                    int grad_idx = ((b * out_channels + f) * h_out + r) * w_out + c;
                                    grad_acc += input_tensor[src_idx] * grad_output[grad_idx];
                                }
                            }
                        }
                        int out_w_idx = ((f * in_channels + ch) * k_dim + kr) * k_dim + kc;
                        grad_kernel[out_w_idx] = grad_acc;
                    }
                }
            }
        }
    }
}

int main() {
    const int NUM_BATCHES = 1;
    const int IN_DEPTH = 1;
    const int IMG_H = 5;
    const int IMG_W = 5;
    const int FILTER_SIZE = 3;
    const int OUT_DEPTH = 1;

    const int MAP_H = IMG_H - FILTER_SIZE + 1;
    const int MAP_W = IMG_W - FILTER_SIZE + 1;

    size_t bytes_src = NUM_BATCHES * IN_DEPTH * IMG_H * IMG_W;
    size_t bytes_filt = OUT_DEPTH * IN_DEPTH * FILTER_SIZE * FILTER_SIZE;
    size_t bytes_dst = NUM_BATCHES * OUT_DEPTH * MAP_H * MAP_W;

    vector<float> vec_input(bytes_src, 1.0f);
    vector<float> vec_filter(bytes_filt, 0.5f);
    vector<float> vec_result(bytes_dst);
    vector<float> vec_d_loss(bytes_dst, 0.1f);
    vector<float> vec_d_filter(bytes_filt);

    cout << "Source Matrix:" << endl;
    display_matrix(vec_input.data(), IMG_H, IMG_W);

    cout << "Initializing Device..." << endl;
    
    float* dev_ptr = vec_result.data();
    #pragma acc parallel loop
    for(int i = 0; i < 10; ++i) {
        dev_ptr[0] = 0.0f;
    }

    auto time_start = chrono::high_resolution_clock::now();
    
    convolution_forward(vec_input.data(), vec_filter.data(), vec_result.data(),
                        NUM_BATCHES, IN_DEPTH, OUT_DEPTH, IMG_H, IMG_W, FILTER_SIZE, MAP_H, MAP_W);

    auto time_mid = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_fwd = time_mid - time_start;
    cout << "FWD Time: " << duration_fwd.count() << " ms" << endl;

    cout << "Result Map:" << endl;
    display_matrix(vec_result.data(), MAP_H, MAP_W);

    auto time_resume = chrono::high_resolution_clock::now();

    convolution_backward(vec_input.data(), vec_d_loss.data(), vec_d_filter.data(),
                         NUM_BATCHES, IN_DEPTH, OUT_DEPTH, IMG_H, IMG_W, FILTER_SIZE, MAP_H, MAP_W);

    auto time_end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_bwd = time_end - time_resume;
    cout << "BWD Time: " << duration_bwd.count() << " ms" << endl;

    cout << "Gradient In:" << endl;
    display_matrix(vec_d_loss.data(), MAP_H, MAP_W);

    cout << "Gradient Kernel Out:" << endl;
    display_matrix(vec_d_filter.data(), FILTER_SIZE, FILTER_SIZE);

    return 0;
}
