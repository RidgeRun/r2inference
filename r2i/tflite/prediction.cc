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

#include "r2i/tflite/prediction.h"

#include <stdio.h>
#include <string.h>

namespace r2i {
namespace tflite {

Prediction::Prediction () {
}

Prediction::~Prediction () {
  this->results_data.clear();
  this->results_sizes.clear();
}

RuntimeError Prediction::AddResults (float *data, unsigned int size) {
  RuntimeError error;

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
  memcpy(internal_data, data, size * sizeof(float));

  this->results_data.push_back(std::shared_ptr<float>(internal_data, free));

  this->results_sizes.push_back(size);

  return error;
}

double Prediction::At (unsigned int output_index, unsigned int index,
                       r2i::RuntimeError &error) {
  error.Clean ();

  if (output_index >= this->results_data.size()) {
    error.Set (RuntimeError::Code::MEMORY_ERROR, "Output index out of bounds");
    return 0;
  }

  if (nullptr == this->results_data[output_index]
      or 0 == this->results_sizes[output_index]) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Prediction was not properly initialized");
    return 0;
  }

  unsigned int n_results =  this->GetResultSize(output_index, error);
  if (RuntimeError::Code::EOK != error.GetCode()) {
    return 0;
  }
  if (n_results < index ) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Triying to access an non-existing index");
    return 0;
  }

  float *fdata = static_cast<float *>(this->GetResultData(output_index, error));
  if (RuntimeError::Code::EOK != error.GetCode()) {
    return 0;
  }
  if (nullptr == fdata) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Prediction result not set yet");
    return 0;
  }

  return fdata[index];
}

void *Prediction::GetResultData (unsigned int output_index,
                                 RuntimeError &error) {
  if (output_index >= this->results_data.size()) {
    error.Set(RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
              "Output index out of bounds");
    return 0;
  }

  return this->results_data[output_index].get();
}

unsigned int Prediction::GetResultSize (unsigned int output_index,
                                        RuntimeError &error) {
  if (output_index >= this->results_sizes.size()) {
    error.Set(RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
              "Output index out of bounds");
    return 0;
  }

  return this->results_sizes[output_index];
}
}
}
