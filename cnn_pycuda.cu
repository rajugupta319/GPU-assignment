import numpy as np
import pycuda.driver as cuda_driver
import pycuda.autoinit
from pycuda.compiler import SourceModule
import tensorflow as tf
import time

cuda_source_code = """
#include <math.h>

__global__ void convolution_forward(float *input_map, float *filters, float *output_map, int batch_size, int in_channels, int out_channels, int in_height, int in_width, int filter_dim, int out_height, int out_width) {
    int batch_id = blockIdx.z;
    int out_ch_id = blockIdx.y;
    int row_out = blockIdx.x / out_width;
    int col_out = blockIdx.x % out_width;

    if (batch_id < batch_size && out_ch_id < out_channels && row_out < out_height && col_out < out_width) {
        float pixel_sum = 0.0f;
        for (int c = 0; c < in_channels; ++c) {
            for (int i = 0; i < filter_dim; ++i) {
                for (int j = 0; j < filter_dim; ++j) {
                    int row_in = row_out + i;
                    int col_in = col_out + j;
                    if (row_in >= 0 && row_in < in_height && col_in >= 0 && col_in < in_width) {
                        int input_idx = ((batch_id * in_channels + c) * in_height + row_in) * in_width + col_in;
                        int filter_idx = ((out_ch_id * in_channels + c) * filter_dim + i) * filter_dim + j;
                        pixel_sum += input_map[input_idx] * filters[filter_idx];
                    }
                }
            }
        }
        int out_idx = ((batch_id * out_channels + out_ch_id) * out_height + row_out) * out_width + col_out;
        output_map[out_idx] = pixel_sum;
    }
}

__global__ void activation_relu(float *data_array, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) data_array[idx] = fmaxf(0.0f, data_array[idx]);
}

__global__ void max_pooling_layer(float *input_data, float *pooled_data, int *max_indices, int num_batches, int num_channels, int h_orig, int w_orig, int h_pool, int w_pool, int pool_stride) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = num_batches * num_channels * h_pool * w_pool;

    if (global_id < total_threads) {
        int ch_id = (global_id / (w_pool * h_pool)) % num_channels;
        int batch_id = global_id / (w_pool * h_pool * num_channels);
        int w_curr = global_id % w_pool;
        int h_curr = (global_id / w_pool) % h_pool;

        int row_start = h_curr * pool_stride;
        int col_start = w_curr * pool_stride;
        int row_end = min(row_start + pool_stride, h_orig);
        int col_end = min(col_start + pool_stride, w_orig);

        float max_val = -1e30f;
        int index_at_max = -1;

        for (int r = row_start; r < row_end; ++r) {
            for (int c = col_start; c < col_end; ++c) {
                int flat_index = ((batch_id * num_channels + ch_id) * h_orig + r) * w_orig + c;
                float current_val = input_data[flat_index];
                if (current_val > max_val) {
                    max_val = current_val;
                    index_at_max = flat_index;
                }
            }
        }
        pooled_data[global_id] = max_val;
        max_indices[global_id] = index_at_max;
    }
}

__global__ void dense_forward_pass(float *input_vec, float *weights, float *bias, float *output_vec, int batch_size, int input_dim, int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_dim) {
        float dot_prod = 0.0f;
        for (int k = 0; k < input_dim; ++k) {
            dot_prod += input_vec[row * input_dim + k] * weights[k * output_dim + col];
        }
        output_vec[row * output_dim + col] = dot_prod + bias[col];
    }
}

__global__ void activation_sigmoid(float *array, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) array[i] = 1.0f / (1.0f + expf(-array[i]));
}

__global__ void calculate_dense_weight_gradients(float *activations, float *errors, float *weight_grads, int batch_size, int input_dim, int output_dim) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < input_dim && c < output_dim) {
        float gradient_sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            gradient_sum += activations[i * input_dim + r] * errors[i * output_dim + c];
        }
        weight_grads[r * output_dim + c] = gradient_sum;
    }
}

__global__ void calculate_dense_bias_gradients(float *errors, float *bias_grads, int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_dim) {
        float bias_sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            bias_sum += errors[i * output_dim + idx];
        }
        bias_grads[idx] = bias_sum;
    }
}
"""

class CudaNeuralNet:
    def __init__(self):
        self.module = SourceModule(cuda_source_code)
        self.kernel_conv = self.module.get_function("convolution_forward")
        self.kernel_relu = self.module.get_function("activation_relu")
        self.kernel_pool = self.module.get_function("max_pooling_layer")
        self.kernel_dense = self.module.get_function("dense_forward_pass")
        self.kernel_sigmoid = self.module.get_function("activation_sigmoid")
        self.kernel_grad_w = self.module.get_function("calculate_dense_weight_gradients")
        self.kernel_grad_b = self.module.get_function("calculate_dense_bias_gradients")

    def train_model(self, input_data, target_labels, total_epochs=1, batch_size=32, learning_rate=0.01):
        print("System Initialized. Starting Training Logic...")
        
        num_samples, img_h, img_w, img_channels = input_data.shape
        filter_dim = 3
        num_feature_maps = 4
        
        feat_h = img_h - filter_dim + 1
        feat_w = img_w - filter_dim + 1
        pool_h = feat_h // 2
        pool_w = feat_w // 2
        
        flattened_size = num_feature_maps * pool_h * pool_w
        
        host_filters = np.random.randn(num_feature_maps, img_channels, filter_dim, filter_dim).astype(np.float32) * 0.1
        host_weights = np.random.randn(flattened_size, 1).astype(np.float32) * 0.1
        host_bias = np.zeros(1).astype(np.float32)

        dev_filters = cuda_driver.mem_alloc(host_filters.nbytes)
        cuda_driver.memcpy_htod(dev_filters, host_filters)
        
        dev_weights = cuda_driver.mem_alloc(host_weights.nbytes)
        cuda_driver.memcpy_htod(dev_weights, host_weights)
        
        dev_bias = cuda_driver.mem_alloc(host_bias.nbytes)
        cuda_driver.memcpy_htod(dev_bias, host_bias)

        for epoch_idx in range(total_epochs):
            time_start = time.time()
            total_error = 0
            correct_predictions = 0
            samples_processed = 0
            
            batches_per_epoch = num_samples // batch_size
            
            for step_idx in range(batches_per_epoch):
                start_idx = step_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_input = input_data[start_idx:end_idx].astype(np.float32).transpose(0, 3, 1, 2).copy()
                batch_targets = target_labels[start_idx:end_idx].astype(np.float32).reshape(-1, 1)
                
                dev_input_batch = cuda_driver.mem_alloc(batch_input.nbytes)
                cuda_driver.memcpy_htod(dev_input_batch, batch_input)
                
                dev_feature_maps = cuda_driver.mem_alloc(batch_size * num_feature_maps * feat_h * feat_w * 4)
                
                self.kernel_conv(dev_input_batch, dev_filters, dev_feature_maps, 
                          np.int32(batch_size), np.int32(img_channels), np.int32(num_feature_maps),
                          np.int32(img_h), np.int32(img_w), np.int32(filter_dim),
                          np.int32(feat_h), np.int32(feat_w),
                          block=(1, 1, 1), grid=(feat_h * feat_w, num_feature_maps, batch_size))
                
                total_elements_map = batch_size * num_feature_maps * feat_h * feat_w
                self.kernel_relu(dev_feature_maps, np.int32(total_elements_map), 
                                 block=(256, 1, 1), grid=((total_elements_map + 255)//256, 1))
                
                dev_pooled_output = cuda_driver.mem_alloc(batch_size * num_feature_maps * pool_h * pool_w * 4)
                dev_pool_indices = cuda_driver.mem_alloc(batch_size * num_feature_maps * pool_h * pool_w * 4)
                total_pooled_elements = batch_size * num_feature_maps * pool_h * pool_w
                
                self.kernel_pool(dev_feature_maps, dev_pooled_output, dev_pool_indices,
                          np.int32(batch_size), np.int32(num_feature_maps),
                          np.int32(feat_h), np.int32(feat_w),
                          np.int32(pool_h), np.int32(pool_w), np.int32(2),
                          block=(256, 1, 1), grid=((total_pooled_elements + 255)//256, 1))

                dev_final_activations = cuda_driver.mem_alloc(batch_size * 4)
                self.kernel_dense(dev_pooled_output, dev_weights, dev_bias, dev_final_activations,
                        np.int32(batch_size), np.int32(flattened_size), np.int32(1),
                        block=(1, 1, 1), grid=(1, batch_size))
                
                self.kernel_sigmoid(dev_final_activations, np.int32(batch_size), block=(1, 1, 1), grid=(1, batch_size))
                
                host_results = np.zeros((batch_size, 1), dtype=np.float32)
                cuda_driver.memcpy_dtoh(host_results, dev_final_activations)
                
                predictions = (host_results > 0.5).astype(np.float32)
                correct_predictions += np.sum(predictions == batch_targets)
                samples_processed += batch_size
                
                host_results = np.clip(host_results, 1e-7, 1.0 - 1e-7)
                batch_cost = -np.mean(batch_targets * np.log(host_results) + (1 - batch_targets) * np.log(1 - host_results))
                total_error += batch_cost

                error_diff = (host_results - batch_targets) / batch_size
                dev_error_diff = cuda_driver.mem_alloc(error_diff.nbytes)
                cuda_driver.memcpy_htod(dev_error_diff, error_diff)
                
                dev_grad_weights = cuda_driver.mem_alloc(host_weights.nbytes)
                dev_grad_bias = cuda_driver.mem_alloc(host_bias.nbytes)
                
                self.kernel_grad_w(dev_pooled_output, dev_error_diff, dev_grad_weights,
                           np.int32(batch_size), np.int32(flattened_size), np.int32(1),
                           block=(1, 16, 1), grid=(1, (flattened_size + 15)//16))
                           
                self.kernel_grad_b(dev_error_diff, dev_grad_bias, np.int32(batch_size), np.int32(1), block=(1,1,1), grid=(1,1))
                
                update_w = np.zeros_like(host_weights)
                update_b = np.zeros_like(host_bias)
                cuda_driver.memcpy_dtoh(update_w, dev_grad_weights)
                cuda_driver.memcpy_dtoh(update_b, dev_grad_bias)
                
                host_weights -= learning_rate * update_w
                host_bias -= learning_rate * update_b
                
                cuda_driver.memcpy_htod(dev_weights, host_weights)
                cuda_driver.memcpy_htod(dev_bias, host_bias)
                
            avg_epoch_error = total_error / batches_per_epoch
            accuracy_percentage = correct_predictions / samples_processed * 100.0
            print(f"Epoch {epoch_idx+1}/{total_epochs} | Loss: {avg_epoch_error:.4f} | Accuracy: {accuracy_percentage:.2f}% | Time: {time.time()-time_start:.2f}s")
        
        print(f"Training Complete.")
        print(f"Final Loss: {avg_epoch_error:.4f}")
        print(f"Final Accuracy: {accuracy_percentage:.2f}%")
        
        test_label = batch_targets[0, 0]
        test_probability = host_results[0, 0]
        test_class = 1.0 if test_probability > 0.5 else 0.0
        
        preview_image = batch_input[0, 0]
        
        print(f"\nSingle Sample Visualization (28x28):")
        
        for r in range(28):
            row_str = ""
            for c in range(28):
                if preview_image[r, c] > 0.5:
                    row_str += "##"
                else:
                    row_str += "  "
            print(row_str)

        print(f"\nGround Truth: {test_label}")
        print(f"Model Probability: {test_probability:.4f}")
        print(f"Predicted Class: {test_class}")

def load_mnist_binary():
    print("Loading MNIST Dataset (Binary Filter: 0 and 1)...")
    (train_x, train_y), _ = tf.keras.datasets.mnist.load_data()
    binary_mask = np.isin(train_y, [0, 1])
    filtered_x = train_x[binary_mask].astype(np.float32) / 255.0
    filtered_x = np.expand_dims(filtered_x, -1)
    filtered_y = train_y[binary_mask]
    print(f"Dataset Size: {len(filtered_x)} samples")
    return filtered_x, filtered_y

if __name__ == "__main__":
    images, labels = load_mnist_binary()
    neural_net = CudaNeuralNet()
    neural_net.train_model(images, labels, total_epochs=2, batch_size=64)
