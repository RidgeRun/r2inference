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

#include "r2i/predictiondecorator.h"

namespace r2i {

PredictionDecorator::PredictionDecorator() {
  this->base_prediction = nullptr;
}

PredictionDecorator::PredictionDecorator(std::shared_ptr<IPrediction> base) {
  this->base_prediction = base;
}

PredictionDecorator::~PredictionDecorator() {
  this->base_prediction = nullptr;
}

double PredictionDecorator::At(unsigned int output_index, unsigned int index,
                               r2i::RuntimeError &error) {
  if (nullptr != this->base_prediction) {
    return this->base_prediction->At(output_index, index, error);
  }

  error.Set (RuntimeError::Code::WRONG_API_USAGE, "Null base prediction");
  return 0;
}

void *PredictionDecorator::GetResultData(unsigned int output_index,
    RuntimeError &error) {
  if (nullptr != this->base_prediction) {
    return this->base_prediction->GetResultData(output_index, error);
  }

  error.Set (RuntimeError::Code::WRONG_API_USAGE, "Null base prediction");
  return nullptr;
}

unsigned int PredictionDecorator::GetResultSize(unsigned int output_index,
    RuntimeError &error) {
  if (nullptr != this->base_prediction) {
    return this->base_prediction->GetResultSize(output_index, error);
  }
  error.Set (RuntimeError::Code::WRONG_API_USAGE, "Null base prediction");
  return 0;
}

RuntimeError PredictionDecorator::AddResult(float *data, unsigned int size) {
  if (nullptr != this->base_prediction) {
    return this->base_prediction->AddResult(data, size);
  }

  RuntimeError error;
  error.Set (RuntimeError::Code::WRONG_API_USAGE, "Null base prediction");
  return error;
}

RuntimeError PredictionDecorator::InsertResult(unsigned int output_index,
    float *data, unsigned int size) {
  if (nullptr != this->base_prediction) {
    return this->base_prediction->InsertResult(output_index, data, size);
  }

  RuntimeError error;
  error.Set (RuntimeError::Code::WRONG_API_USAGE, "Null base prediction");
  return error;
}

unsigned int PredictionDecorator::GetOutputCount() {
  if (nullptr != this->base_prediction) {
    return this->base_prediction->GetOutputCount();
  }

  return 0;
}

}