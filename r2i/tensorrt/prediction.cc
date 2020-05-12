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


RuntimeError Prediction::SetResultBuffer (std::shared_ptr<void> result_buffer,
    size_t num, DataType data_type) {
  cudaError_t cuda_error;
  RuntimeError error;
  std::shared_ptr<void> buff;

  buff = std::shared_ptr<void>(malloc(num * data_type.GetBytesPerPixel()), free);
  if (!buff) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Unable to allocate memory for result");
    return error;
  }

  if (nullptr == result_buffer) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid result buffer passed to prediction");
    return error;
  }

  cuda_error = cudaMemcpy(buff.get(),
                          result_buffer.get(),
                          num * data_type.GetBytesPerPixel(),
                          cudaMemcpyDeviceToHost);
  if (cudaSuccess != cuda_error) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Unable to read data from CUDA");
    return error;
  }

  this->result_buffer = std::shared_ptr<void>(buff);
  this->num = num;
  this->data_type = data_type;

  return error;
}

unsigned int Prediction::GetResultSize () {
  return this->num * this->data_type.GetBytesPerPixel();
}

void *Prediction::GetResultData () {
  if (nullptr == this->result_buffer) {
    return nullptr;
  }

  return result_buffer.get();
}

template <class T>
static double PointerToResult (void *buff, unsigned int index,
                               RuntimeError &error) {
  T *data = static_cast<T *>(buff);

  if (nullptr == data) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Prediction result not set yet");
    return 0;
  }

  return data[index];
}

double Prediction::At (unsigned int index, RuntimeError &error) {
  error.Clean ();
  double result;

  if (nullptr == this->result_buffer) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Prediction was not properly initialized");
    return 0;
  }

  unsigned int n_results =  this->num;
  if ( n_results < index ) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Trying to access an non-existing index");
    return 0;
  }

  switch ( this->data_type.GetId() ) {
    case DataType::FLOAT :
      result = PointerToResult<float>( this->GetResultData(), index, error);
      break;
    default :
      result = 0;
      error.Set (RuntimeError::Code::WRONG_API_USAGE,
                 "Unsupported data type");
  }

  return result;
}

}
}
