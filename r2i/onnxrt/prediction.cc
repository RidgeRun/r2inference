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

#include "r2i/onnxrt/prediction.h"

#include <cstring>

namespace r2i {
namespace onnxrt {

Prediction::Prediction ():
  output_data(nullptr), tensor_size(0) {
}

Prediction::~Prediction () {
}

RuntimeError Prediction::SetTensorValues(float *output_data, int data_size) {
  RuntimeError error;

  if (nullptr == output_data) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid output tensor values passed to prediction");
    return error;
  }

  if (0 == data_size) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor size passed to prediction");
    return error;
  }

  this->tensor_size = data_size;

  this->output_data = std::shared_ptr<float>(new float[this->tensor_size],
                      std::default_delete<float[]>());

  std::memcpy(this->output_data.get(), output_data,
              this->tensor_size * sizeof(float));

  return error;
}

double Prediction::At (unsigned int index,  r2i::RuntimeError &error) {
  error.Clean ();

  if (nullptr == this->output_data) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Prediction was not properly initialized");
    return 0;
  }

  unsigned int n_results =  this->GetResultSize();
  if (n_results < index) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Triying to access an non-existing index");
    return 0;
  }
  return this->output_data.get()[index];
}

void *Prediction::GetResultData () {
  return reinterpret_cast<void *>(this->output_data.get());
}

unsigned int Prediction::GetResultSize () {
  return this->tensor_size;
}

}  // namespace onnxrt
}  // namespace r2i
