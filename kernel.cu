#include <iostream>
#include "plugin.hpp"

const int THREAD_NUM = 256;
inline int GET_BLOCKS(const int count){
    return (count + THREAD_NUM - 1) / THREAD_NUM; 
}



#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__device__ inline float get_pixel(float* const img, int x, int y, int c,const int w,const int h){
    //img in format HWC
    if (x < 0 || x >= w || y < 0 || y >= h || c < 0 || c >= 3) return 0;
    return img[(y*w+x)*3 + 2-c];
}

__global__ void preprocess_kernel(const int nthreads, float* const bottom_data, float * const output){
    int indexs[4]; //NCHW
    const int num_axes = 4;
    const int step = RESIZE_H * RESIZE_W;
    const int old_steps[4] = {CHANNEL*ORIGIN_H*ORIGIN_W, CHANNEL*ORIGIN_W, CHANNEL, 1};
    const int new_steps[4] = {CHANNEL*RESIZE_H*RESIZE_W, RESIZE_H*RESIZE_W, RESIZE_W, 1};
    const float mean_values[3] = {0.,0.,0.};
    /*const float mean_values[3] = {103.94,116.78,123.68};*/
    const float wscale = ((float)ORIGIN_W)/RESIZE_W;
    const float hscale = ((float)ORIGIN_H)/RESIZE_H;
    /*const float scale = 0.017;*/
    const float scale = 1./255;

    CUDA_KERNEL_LOOP(index, nthreads) {
        int temp_idx = index;
        
        for (int i = 0; i < num_axes; ++i) {
            indexs[i] = temp_idx / new_steps[i];
            temp_idx %= new_steps[i];
        }
        int mean_idx = index/step;
        //coordinate of origin image
        float x = (float)indexs[3] * wscale;
        float y = (float)indexs[2] * hscale;
        int ix = (int)x;
        int iy = (int)y;
        float dx = x - ix;
        float dy = y - iy;
        float val = (1-dy)*(1-dx)*get_pixel(bottom_data + old_steps[0]*indexs[0], ix, iy, indexs[1],ORIGIN_W, ORIGIN_H)
            + (dy)*(1-dx)*get_pixel(bottom_data+old_steps[0]*indexs[0], ix, iy+1, indexs[1], ORIGIN_W, ORIGIN_H)
            + (1-dy)*(dx)*get_pixel(bottom_data+old_steps[0]*indexs[0], ix+1, iy, indexs[1], ORIGIN_W, ORIGIN_H)
            + (dy)*(dx)*get_pixel(bottom_data+old_steps[0]*indexs[0], ix+1, iy+1, indexs[1], ORIGIN_W, ORIGIN_H);
        output[index] = (val - mean_values[mean_idx]) * scale;
        /*printf("indexs (%d, %d, %d, %d)\n", indexs[0], indexs[1], indexs[2], indexs[3]);*/
        /*printf("pixels (%f, %f, %f, %f)\n",get_pixel(bottom_data + new_steps[0]*indexs[0], ix, iy, indexs[1],ORIGIN_W, ORIGIN_H),*/
                /*get_pixel(bottom_data+new_steps[0]*indexs[0], ix, iy+1, indexs[1], ORIGIN_W, ORIGIN_H),*/
                /*get_pixel(bottom_data+new_steps[0]*indexs[0], ix+1, iy, indexs[1], ORIGIN_W, ORIGIN_H),*/
                /*get_pixel(bottom_data+new_steps[0]*indexs[0], ix+1, iy+1, indexs[1], ORIGIN_W, ORIGIN_H));*/
        /*printf("output[%d]: %f, x(%f) y(%f) ix(%d) iy(%d) val(%f)\n", index, output[index],*/
                /*x, y, ix, iy, val);*/
    }
}

void preprocess(int count, float *input, float* output){
    preprocess_kernel<<<GET_BLOCKS(count), THREAD_NUM>>>(count, input, output);
}


__global__ void reorg_kernel(const int nthreads, float* const input, float *const output, const int width, const int height, const int channels) {
    const int stride = REORG_STRIDE;
    int tc = channels / (stride*stride);
    int tw = width * stride;
    int th = height * stride;
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tmp = index;
        int w = index % width;
        int h = (tmp /= width) % height;
        int c = (tmp /= height) % channels;
        int n = tmp / channels;
        int c2 = c % tc;
        int offset = c / tc;
        int w2 = w * stride + offset % stride;
        int h2 = h * stride + offset / stride;
        int out_idx = w2 + tw * (h2 + th * (c2 + tc * n));
        output[index] = input[out_idx];
    }
}


void reorg(float *input, float* output, int batchSz){
    const int width = RESIZE_W / 16;
    const int height = RESIZE_H / 16;
    const int channels = 64;
    const int count = width * height * channels * batchSz;
    reorg_kernel<<<GET_BLOCKS(count), THREAD_NUM>>>(count, input, output, width, height, channels);
}

__device__ float sigmoid(float x) {
    return 1./(exp(-x) + 1.);
}


__global__ void postprocess_kernel(const int nthreads, const int channels,
                                const int height, const int width, const int num_classes,
                                const int num_box_in_grid, float* input, float* output){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int N = width * height * num_box_in_grid;
        int b = index / N;
        int bbox_index = index % N;
        int grid_index = bbox_index / num_box_in_grid;
        int bbox_in_grid_index = bbox_index % num_box_in_grid;
        int h = grid_index / width;
        int w = grid_index % width;
        int c = bbox_in_grid_index * ( 5 + num_classes );
        int offset = ((((b * channels) + c) * height) + h) *width + w;
        int step = width *height;
        float data[5 + CLASS_NUM];
        
        data[0] = sigmoid(input[offset]);
        data[1] = sigmoid(input[offset + step]);
        data[2] = exp(input[offset + 2 * step]);
        data[3] = exp(input[offset + 3 * step]);
        data[4] = sigmoid(input[offset + 4 * step]);
        data[5] = 1;
        /*for (int i = 0; i < CLASS_NUM; i++) {*/
            /*data[5+i] = input[offset + (5+i)*step];*/
        /*}*/
        
        offset = ((((b*height) + h) * width) + w) * channels + c;
        for (int i = 0; i < 5 + CLASS_NUM; i++) {
            output[offset+i] = data[i];
        }
    }
}

void postprocess(float* input, float* output, int batch) {
    const int width = RESIZE_W / 32;
    const int height = RESIZE_H / 32;
    const int channels = BOX_NUM * (5 + CLASS_NUM);
    const int num_classes = CLASS_NUM;
    const int num_box_in_grid = BOX_NUM;
    const int count =  width * height * num_box_in_grid * batch;
    postprocess_kernel<<<GET_BLOCKS(count), THREAD_NUM>>>(count, channels, height,
                        width, num_classes, num_box_in_grid, input, output);
}

__global__ void leakyrelu_kernel(const int nthreads, float* input, float* output) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        output[index] = (input[index] > 0)? input[index] : 0.1*input[index];
    }
}

void leakyrelu(float* input, float* output, int count) {
    leakyrelu_kernel<<<GET_BLOCKS(count), THREAD_NUM>>>(count, input, output);
}


__global__ void depthwiseConv_kernel(int count, const float* const input, const float*const weight_data,
        int batch, int channels, int top_height, int top_width, const int bottom_height, 
        const int bottom_width, int kernel_h, int kernel_w, const int stride_h,
        const int stride_w, const int pad_h, const int pad_w, float* const output) {
    CUDA_KERNEL_LOOP(index, count) {
/*        const int w = index % top_width;*/
        /*index /= top_width;*/
        /*const int h = index % top_height;*/
        /*index /= top_height;*/
        /*const int c = index % channels;*/
        /*index /= channels;*/
        /*const int n = index;*/
    const int n = index / channels / top_height / top_width;
    const int c = (index / top_height / top_width) % channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
        const float* weight = weight_data + c * kernel_h * kernel_w;
        float value = 0.;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++ kw) {
                const int h_in = -pad_h + h * stride_h + kh;
                const int w_in = -pad_w + w * stride_w + kw;
                if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) &&
                        (w_in < bottom_width)) {
                    const int offset = ((n*channels + c) * bottom_height + h_in) *
                        bottom_width + w_in;
                    value += (*weight) * input[offset];
                }
                ++weight;
            }
        }
        output[index] = value;
    }
}

void depthwiseConv(int batch, int c, int w, int h, int stride, float* input, float* weight, float* output) {
    const int kernel_size = KERNEL_SIZE;
    const int pad = KERNEL_SIZE / 2;
    const int top_height = (h + 2*pad - kernel_size) / stride + 1;
    const int top_width = (w + 2*pad - kernel_size) / stride + 1;
    const int count = c*top_height*top_width;
    /*printf("top CHW:(%d %d %d)", c, top_height, top_width);*/
    depthwiseConv_kernel<<<GET_BLOCKS(count), THREAD_NUM>>>(count,input, weight, batch, 
            c, top_height, top_width, h, w, kernel_size, kernel_size, stride, 
            stride, pad, pad, output);

}

