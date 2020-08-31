/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include "r2i/tensorrt/prediction.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace r2i {
namespace tensorrt {

Prediction::Prediction () {
}


RuntimeError Prediction::AddResult (float *data, unsigned int size) {
  RuntimeError error;
  cudaError_t cuda_error;

  if (nullptr == data) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor values passed to prediction");
    return error;
  }

  if (0 == size) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor size passed to prediction");
    return error;
  }

  float *internal_data = (float *)malloc(size * sizeof(float));
  if (nullptr == internal_data) {
    error.Set(RuntimeError::Code::MEMORY_ERROR,
              "Error while allocating prediction result memory.");
    return error;
  }

  cuda_error = cudaMemcpy(internal_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaSuccess != cuda_error) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Unable to read data from CUDA");
    return error;
  }

  this->results_data.push_back(std::shared_ptr<float>(internal_data, free));
  this->results_sizes.push_back(size);

  return error;
}

RuntimeError Prediction::InsertResult(unsigned int output_index, float *data,
                                      unsigned int size) {
  RuntimeError error;
  cudaError_t cuda_error;

  if (output_index >= this->results_data.size()) {
    error.Set(RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
              "Output index out of bounds");
    return error;
  }

  if (nullptr == data) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor values passed to prediction");
    return error;
  }

  if (0 == size) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor size passed to prediction");
    return error;
  }

  float *internal_data = this->results_data[output_index].get();

  cuda_error = cudaMemcpy(internal_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaSuccess != cuda_error) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Unable to read data from CUDA");
    return error;
  }

  return error;
}

}
}
