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

int64_t Prediction::GetRequiredBufferSize (TF_Output output, int64_t *dims,
    int64_t num_dims) {
  int64_t size = 1;

  /* For each dimension, multiply the amount of entries */
  for (int dim = 0; dim < num_dims; ++dim) {
    size *= dims[dim];
  }

  return size;
}

RuntimeError Prediction::SetTensor (std::shared_ptr<TF_Graph> pgraph,
                                    TF_Operation *operation, std::shared_ptr<TF_Tensor> tensor) {
  RuntimeError error;

  if (nullptr == pgraph) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid graph passed to prediction");
    return error;
  }

  if (nullptr == operation) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid operation passed to prediction");
    return error;
  }

  if (nullptr == tensor) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor passed to prediction");
    return error;
  }

  std::shared_ptr<TF_Status> pstatus (TF_NewStatus(), TF_DeleteStatus);
  TF_Status *status = pstatus.get ();
  TF_Graph *graph = pgraph.get ();
  TF_Output output = { .oper = operation, .index = 0 };

  int num_dims = TF_GraphGetTensorNumDims(graph, output, status);
  if (TF_GetCode(status) != TF_OK) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
    return error;
  }

  int64_t dims[num_dims];
  TF_GraphGetTensorShape(graph, output, dims, num_dims, status);
  if (TF_GetCode(status) != TF_OK) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
    return error;
  }

  /* R2Inference uses a batch size of 1 but some tensors have this value set to
   * generic (-1) or greater than 1.
   * Batch size set to 1 for general compatibility support. */
  dims[0] = 1;

  TF_DataType type = TF_OperationOutputType(output);
  size_t type_size = TF_DataTypeSize(type);
  size_t data_size = this->GetRequiredBufferSize (output, dims, num_dims);

  if (TF_FLOAT != type) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The output of this model is not floating point");
    return error;
  }

  this->tensor = tensor;
  this->result_size = data_size * type_size;

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
