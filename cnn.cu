#include <iostream>
#include <vector>
#include <cuda_runtime.h>


using namespace std;


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void convolution_forward_pass(
    const float* __restrict__ input_tensor, 
    const float* __restrict__ kernel_weights, 
    float* __restrict__ output_map, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int in_h, 
    int in_w, 
    int k_size, 
    int out_h, 
    int out_w) 
{

    int batch_id = blockIdx.z;
    int filter_id = blockIdx.y;
    int pixel_id = blockIdx.x * blockDim.x + threadIdx.x;

    int row_out = pixel_id / out_w;
    int col_out = pixel_id % out_w;


    if (batch_id < batch_size && filter_id < out_channels && row_out < out_h && col_out < out_w) {
        float pixel_sum = 0.0f;

      
        for (int c = 0; c < in_channels; ++c) {
            for (int i = 0; i < k_size; ++i) {     
                for (int j = 0; j < k_size; ++j) { 
                    
                    int current_h = row_out + i;
                    int current_w = col_out + j;
                  
                    int input_idx = ((batch_id * in_channels + c) * in_h + current_h) * in_w + current_w;
                    int weight_idx = ((filter_id * in_channels + c) * k_size + i) * k_size + j;
                    
                    pixel_sum += input_tensor[input_idx] * kernel_weights[weight_idx];
                }
            }
        }
        
        int output_idx = ((batch_id * out_channels + filter_id) * out_h + row_out) * out_w + col_out;
        output_map[output_idx] = pixel_sum;
    }
}

__global__ void compute_weight_gradients(
    const float* __restrict__ input_tensor, 
    const float* __restrict__ output_grads, 
    float* __restrict__ weight_grads, 
    int batch_size, 
    int in_channels, 
    int out_channels, 
    int in_h, 
    int in_w, 
    int k_size, 
    int out_h, 
    int out_w) 
{

    int filter_id = blockIdx.y;
    int channel_id = blockIdx.x;
    
   
    int tid = threadIdx.x;
    int k_row = tid / k_size;
    int k_col = tid % k_size;

    if (filter_id < out_channels && channel_id < in_channels && k_row < k_size && k_col < k_size) {
        float gradient_accum = 0.0f;

        for (int b = 0; b < batch_size; ++b) {
            for (int r = 0; r < out_h; ++r) {
                for (int c = 0; c < out_w; ++c) {
                    
                    int input_r = r + k_row;
                    int input_c = c + k_col;
              
                    int in_idx = ((b * in_channels + channel_id) * in_h + input_r) * in_w + input_c;
                    int grad_idx = ((b * out_channels + filter_id) * out_h + r) * out_w + c;
                    
                    gradient_accum += input_tensor[in_idx] * output_grads[grad_idx];
                }
            }
        }
        

        int weight_out_idx = ((filter_id * in_channels + channel_id) * k_size + k_row) * k_size + k_col;
        weight_grads[weight_out_idx] = gradient_accum;
    }
}

int main() {
 
    const int NUM_BATCHES = 1;
    const int INPUT_CHANNELS = 1; 
    const int HEIGHT = 5;
    const int WIDTH = 5;
    const int KERNEL_DIM = 3;
    const int OUTPUT_CHANNELS = 1;
    
    const int OUT_HEIGHT = HEIGHT - KERNEL_DIM + 1;
    const int OUT_WIDTH = WIDTH - KERNEL_DIM + 1;

   
    size_t bytes_input = NUM_BATCHES * INPUT_CHANNELS * HEIGHT * WIDTH * sizeof(float);
    size_t bytes_weights = OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_DIM * KERNEL_DIM * sizeof(float);
    size_t bytes_output = NUM_BATCHES * OUTPUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH * sizeof(float);

    vector<float> host_input(NUM_BATCHES * INPUT_CHANNELS * HEIGHT * WIDTH, 1.0f);
    vector<float> host_weights(OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_DIM * KERNEL_DIM, 0.5f);
    vector<float> host_grad_incoming(NUM_BATCHES * OUTPUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH, 0.1f);
   
    vector<float> host_output_result(NUM_BATCHES * OUTPUT_CHANNELS * OUT_HEIGHT * OUT_WIDTH);
    vector<float> host_weight_grad_result(OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_DIM * KERNEL_DIM);

    cout << "=== Input Matrix Preview ===" << endl;
    for(int i = 0; i < HEIGHT; ++i) {
        for(int j = 0; j < WIDTH; ++j) {
            cout << host_input[i * WIDTH + j] << " ";
        }
        cout << endl;
    }

    float *d_input, *d_weights, *d_output, *d_grad_in, *d_grad_weights;
    CUDA_CHECK(cudaMalloc(&d_input, bytes_input));
    CUDA_CHECK(cudaMalloc(&d_weights, bytes_weights));
    CUDA_CHECK(cudaMalloc(&d_output, bytes_output));
    CUDA_CHECK(cudaMalloc(&d_grad_in, bytes_output));
    CUDA_CHECK(cudaMalloc(&d_grad_weights, bytes_weights));

  
    CUDA_CHECK(cudaMemcpy(d_input, host_input.data(), bytes_input, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, host_weights.data(), bytes_weights, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_in, host_grad_incoming.data(), bytes_output, cudaMemcpyHostToDevice));

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    dim3 block_size(256);
    dim3 grid_size((OUT_HEIGHT * OUT_WIDTH + 255) / 256, OUTPUT_CHANNELS, NUM_BATCHES);
    
    cudaEventRecord(start_event);
    convolution_forward_pass<<<grid_size, block_size>>>(
        d_input, d_weights, d_output, 
        NUM_BATCHES, INPUT_CHANNELS, OUTPUT_CHANNELS, 
        HEIGHT, WIDTH, KERNEL_DIM, OUT_HEIGHT, OUT_WIDTH
    );
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    cout << "Forward Pass Duration: " << elapsed_ms << " ms" << endl;

    CUDA_CHECK(cudaMemcpy(host_output_result.data(), d_output, bytes_output, cudaMemcpyDeviceToHost));
    
    cout << "=== Forward Output ===" << endl;
    for(int i = 0; i < OUT_HEIGHT; ++i) {
        for(int j = 0; j < OUT_WIDTH; ++j) {
            cout << host_output_result[i * OUT_WIDTH + j] << " ";
        }
        cout << endl;
    }


    dim3 back_block(KERNEL_DIM * KERNEL_DIM);
    dim3 back_grid(INPUT_CHANNELS, OUTPUT_CHANNELS);
    
    cudaEventRecord(start_event);
    compute_weight_gradients<<<back_grid, back_block>>>(
        d_input, d_grad_in, d_grad_weights, 
        NUM_BATCHES, INPUT_CHANNELS, OUTPUT_CHANNELS, 
        HEIGHT, WIDTH, KERNEL_DIM, OUT_HEIGHT, OUT_WIDTH
    );
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    cout << "Gradient Calculation Duration: " << elapsed_ms << " ms" << endl;

    CUDA_CHECK(cudaMemcpy(host_weight_grad_result.data(), d_grad_weights, bytes_weights, cudaMemcpyDeviceToHost));
    
    cout << "=== Incoming Gradients (dY) ===" << endl;
    for(int i = 0; i < OUT_HEIGHT; ++i) {
        for(int j = 0; j < OUT_WIDTH; ++j) {
            cout << host_grad_incoming[i * OUT_WIDTH + j] << " ";
        }
        cout << endl;
    }

    cout << "=== Calculated Weight Gradients (dW) ===" << endl;
    for(int i = 0; i < KERNEL_DIM; ++i) {
        for(int j = 0; j < KERNEL_DIM; ++j) {
            cout << host_weight_grad_result[i * KERNEL_DIM + j] << " ";
        }
        cout << endl;
    }
    
 
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    cudaFree(d_grad_in);
    cudaFree(d_grad_weights);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
