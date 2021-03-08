/* Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include "r2i/tensorflow/prediction.h"

#include <tensorflow/c/c_api.h>

namespace r2i {
namespace tensorflow {

Prediction::Prediction () :
  tensor(nullptr), result_size(0) {
}

RuntimeError Prediction::SetTensor (std::shared_ptr<TF_Tensor> tensor) {
  RuntimeError error;

  if (nullptr == tensor) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor passed to prediction");
    return error;
  }

  this->tensor = tensor;
  this->result_size = TF_TensorByteSize (tensor.get());

  return error;
}

unsigned int Prediction::GetResultSize () {
  return this->result_size;
}

void *Prediction::GetResultData () {
  if (nullptr == this->tensor) {
    return nullptr;
  }

  return TF_TensorData(this->tensor.get());
}

double Prediction::At (unsigned int index,  RuntimeError &error) {
  error.Clean ();

  if (nullptr == this->tensor) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Prediction was not properly initialized");
    return 0;
  }

  unsigned int n_results =  this->GetResultSize() / sizeof(float);
  if (n_results < index ) {
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
