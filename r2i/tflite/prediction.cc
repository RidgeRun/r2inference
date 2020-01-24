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

Prediction::Prediction ():
  outputdata(nullptr), tensorsize(0) {
}
Prediction::~Prediction () {
  if (nullptr != this->outputdata ) {
    free(this->outputdata);
  }
}

RuntimeError Prediction::SetTensorValues(float *outputdata, int tensorsize) {
  RuntimeError error;

  if (nullptr == outputdata) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor values passed to prediction");
    return error;
  }

  if (0 == tensorsize) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Invalid tensor size passed to prediction");
    return error;
  }

  this->tensorsize = tensorsize;

  this->outputdata = (float *) malloc(this->tensorsize * sizeof(float));

  memcpy(this->outputdata, outputdata, sizeof(outputdata) + 1);

  return error;
}

double Prediction::At (unsigned int index,  r2i::RuntimeError &error) {
  error.Clean ();

  if (nullptr == this->outputdata or 0 == this->tensorsize) {
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
  return this->outputdata[index];
}

void *Prediction::GetResultData () {
  return (void *)this->outputdata;
}

unsigned int Prediction::GetResultSize () {
  return this->tensorsize;
}
}
}
