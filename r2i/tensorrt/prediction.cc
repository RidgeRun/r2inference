/* Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
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


RuntimeError Prediction::SetResultBuffer (void *result_buffer, size_t size) {
  cudaError_t cuda_error;
  RuntimeError error;
  std::shared_ptr<void> buff;

  if (nullptr == result_buffer) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid result buffer passed to prediction");
    return error;
  }

  buff = std::shared_ptr<void>(malloc(size), free);
  if (!buff) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Unable to allocate memory for result");
    return error;

  }

  cuda_error = cudaMemcpy(buff.get(),
                          result_buffer,
                          size,
                          cudaMemcpyDeviceToHost);
  if (cudaSuccess != cuda_error) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Unable to read data from CUDA");
    return error;
  }

  this->result_buffer = std::shared_ptr<void>(buff);
  this->result_size = size;

  return error;
}

unsigned int Prediction::GetResultSize () {
  return this->result_size;
}

void *Prediction::GetResultData () {
  if (nullptr == this->result_buffer) {
    return nullptr;
  }

  return result_buffer.get();
}

double Prediction::At (unsigned int index,  RuntimeError &error) {
  error.Clean ();

  if (nullptr == this->result_buffer) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Prediction was not properly initialized");
    return 0;
  }

  /* FIXME get correct datatype */
  unsigned int n_results =  this->GetResultSize() / sizeof(float);
  if ( n_results < index ) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Triying to access an non-existing index");
    return 0;
  }

  float *fdata = static_cast<float *> (this->GetResultData ());
  if (nullptr == fdata) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Prediction result not set yet");
    return 0;
  }

  return fdata[index];
}

}
}
