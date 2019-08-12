/*
* Copyright (c) 2019, NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once
#include "col_condenser.cuh"
#include "cub/cub.cuh"
#include "quantile.h"
#include <float.h>

__global__ void set_sorting_offset(const int nrows, const int ncols,
                                  int *offsets) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid <= ncols) offsets[tid] = tid * nrows;

  return;
}

template <typename T>
__global__ void get_all_quantiles(const T *__restrict__ data, T *quantile,
                                  const int nrows, const int ncols,
                                  const int nbins) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nbins * ncols) {
    int binoff = (int)(nrows / nbins);
    int coloff = (int)(tid / nbins) * nrows;
    quantile[tid] = data[((tid % nbins) + 1) * binoff - 1 + coloff];
  }
  return;
}

template <typename T>
__global__ void get_all_quantiles_w_histogram(const T *__restrict__ histo, T *quantile,
                                  const int nrows, const int ncols,
                                  const float min, const float max, 
                                  const int nbins, const int num_level) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < nbins * ncols) {
    float target = (float)nrows * ((float)tid / (float)nbins);
    float sum = 0;
    float diff = 0;
    float best_for_now = FLT_MAX;

    for (int i = 0; i < num_level; ++i) {
      sum += histo[i]; 
      diff = target - sum > 0 ? target - sum : sum - target;
      if (diff < best_for_now) {
        best_for_now = diff;
        quantile[tid] = ((max - min) / (num_level - 1)) * i + min;
      }
    }
  }
  return;
}

template <typename T, typename L>
void preprocess_quantile(const T *data, const unsigned int *rowids,
                        const int n_sampled_rows, const int ncols,
                        const int rowoffset, const int nbins,
                        std::shared_ptr<TemporaryMemory<T, L>> tempmem) {
  /*
    // Dynamically determine batch_cols (number of columns processed per loop iteration) from the available device memory.
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    int max_ncols = free_mem / (2 * n_sampled_rows * sizeof(T));
    int batch_cols = (max_ncols > ncols) ? ncols : max_ncols;
    ASSERT(max_ncols != 0, "Cannot preprocess quantiles due to insufficient device memory.");
    */
  int batch_cols =
    1;  // Processing one column at a time, for now, until an appropriate getMemInfo function is provided for the deviceAllocator interface.

  int threads = 128;
  MLCommon::device_buffer<int> *d_offsets;
  // MLCommon::device_buffer<T> *d_keys_out;
  const T *d_keys_in;
  int blocks;
  // if (tempmem->temp_data != nullptr) {
  //   T *d_keys_out = tempmem->temp_data->data();
  //   unsigned int *colids = nullptr;
  //   blocks = MLCommon::ceildiv(ncols * n_sampled_rows, threads);
  //   allcolsampler_kernel<<<blocks, threads, 0, tempmem->stream>>>(
  //     data, rowids, colids, n_sampled_rows, ncols, rowoffset,
  //     d_keys_out);  // d_keys_in already allocated for all ncols
  //   CUDA_CHECK(cudaGetLastError());
  //   d_keys_in = d_keys_out;
  // } else {
  //   d_keys_in = data;
  // }
  d_keys_in = data;

  d_offsets = new MLCommon::device_buffer<int>(
    tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, batch_cols + 1);

  blocks = MLCommon::ceildiv(batch_cols + 1, threads);
  set_sorting_offset<<<blocks, threads, 0, tempmem->stream>>>(
    n_sampled_rows, batch_cols, d_offsets->data());
  CUDA_CHECK(cudaGetLastError());

  // Determine temporary device storage requirements
  MLCommon::device_buffer<char> *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  size_t temp_stoage_bytes_max = 0; 
  size_t temp_stoage_bytes_min = 0; 
  size_t temp_stoage_bytes_histo = 0; 

  int batch_cnt =
    MLCommon::ceildiv(ncols, batch_cols);  // number of loop iterations
  int last_batch_size =
    ncols - batch_cols * (batch_cnt - 1);  // number of columns in last batch
  int batch_items =
    n_sampled_rows * batch_cols;  // used to determine d_temp_storage size

  int num_level = 10000; 
  MLCommon::device_buffer<T> *d_max, *d_min, *d_histo;
  std::vector<T> h_max(1);
  std::vector<T> h_min(1);
  d_max = new MLCommon::device_buffer<T>(
    tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, 1);
  d_min = new MLCommon::device_buffer<T>(
    tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, 1);
  d_histo = new MLCommon::device_buffer<T>(
    tempmem->ml_handle.getDeviceAllocator(), tempmem->stream, num_level);
  MLCommon::device_buffer<char> *d_temp_storage_2 = nullptr;

  CUDA_CHECK(cub::DeviceReduce::Max(d_temp_storage, temp_stoage_bytes_max, d_keys_in, d_max->data(), batch_items, tempmem->stream));
  CUDA_CHECK(cub::DeviceReduce::Min(d_temp_storage, temp_stoage_bytes_min, d_keys_in, d_min->data(), batch_items, tempmem->stream));
  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_stoage_bytes_histo,
    d_keys_in, d_histo->data(), num_level, 0, 1, batch_items, tempmem->stream));

  temp_storage_bytes = std::max(std::max(temp_stoage_bytes_max, temp_stoage_bytes_min), temp_stoage_bytes_histo);

  // Allocate temporary storage
  d_temp_storage =
    new MLCommon::device_buffer<char>(tempmem->ml_handle.getDeviceAllocator(),
                                      tempmem->stream, temp_storage_bytes);

  // Compute quantiles for cur_batch_cols columns per loop iteration.
  for (int batch = 0; batch < batch_cnt; batch++) {
    int cur_batch_cols = (batch == batch_cnt - 1)
                          ? last_batch_size
                          : batch_cols;  // properly handle the last batch

    int batch_offset = batch * n_sampled_rows * batch_cols;
    int quantile_offset = batch * nbins * batch_cols;

    CUDA_CHECK(cub::DeviceReduce::Max((void *)d_temp_storage->data(), temp_storage_bytes, &d_keys_in[batch_offset], d_max->data(), batch_items, tempmem->stream));
    CUDA_CHECK(cub::DeviceReduce::Min((void *)d_temp_storage->data(), temp_storage_bytes, &d_keys_in[batch_offset], d_min->data(), batch_items, tempmem->stream));
    
    MLCommon::updateHost<T>(h_min.data(), d_min->data(), 1, tempmem->stream);
    MLCommon::updateHost<T>(h_max.data(), d_max->data(), 1, tempmem->stream);
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

    CUDA_CHECK(cub::DeviceHistogram::HistogramEven((void *)d_temp_storage->data(), temp_storage_bytes, 
      &d_keys_in[batch_offset], d_histo->data(), num_level, h_min[0], h_max[0], batch_items, tempmem->stream));

    blocks = MLCommon::ceildiv(cur_batch_cols * nbins, threads);
    get_all_quantiles_w_histogram<<<blocks, threads, 0, tempmem->stream>>>(
      d_histo->data(), &tempmem->d_quantile->data()[quantile_offset],
      batch_items, cur_batch_cols, h_min[0], h_max[0], nbins, num_level);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
  }
  MLCommon::updateHost(tempmem->h_quantile->data(), tempmem->d_quantile->data(),
                      nbins * ncols, tempmem->stream);
  h_max.clear();
  h_min.clear();
  d_histo->release(tempmem->stream);
  d_offsets->release(tempmem->stream);
  d_temp_storage->release(tempmem->stream);
  delete d_histo;
  delete d_offsets;
  delete d_temp_storage;

  return;
}
