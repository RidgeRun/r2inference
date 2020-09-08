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

#include "r2i/predictionclassification.h"

namespace r2i {

PredictionClassification::PredictionClassification() {
  this->base_prediction = nullptr;
}

PredictionClassification::PredictionClassification(std::shared_ptr<IPrediction>
    base) {
  this->base_prediction = base;
}

PredictionClassification::~PredictionClassification() {
  this->base_prediction = nullptr;
  this->scores.clear();
  this->labels.clear();
}

RuntimeError PredictionClassification::SetScores(float *scores,
    unsigned int size) {
  RuntimeError error;
  if (nullptr == scores) {
    error.Set(RuntimeError::Code::NULL_PARAMETER, "Null scores pointer");
    return error;
  }
  if (0 == size) {
    error.Set(RuntimeError::Code::INCOMPATIBLE_PARAMETERS, "Scores size is zero");
  }

  this->scores = std::vector<float>(scores, scores + (size * sizeof(float)) );
  return error;
}

RuntimeError PredictionClassification::SetLabels(int *labels,
    unsigned int size) {
  RuntimeError error;
  if (nullptr == labels) {
    error.Set(RuntimeError::Code::NULL_PARAMETER, "Null labels pointer");
    return error;
  }
  if (0 == size) {
    error.Set(RuntimeError::Code::INCOMPATIBLE_PARAMETERS, "Scores size is zero");
  }

  this->labels = std::vector<int>(labels, labels + (size * sizeof(int)) );
  return error;
}

float *PredictionClassification::GetScores() {
  return this->scores.data();
}

int *PredictionClassification::GetLabels() {
  return this->labels.data();
}

}