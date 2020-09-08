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

#include "r2i/predictiondetection.h"

namespace r2i {

PredictionDetection::PredictionDetection() {
  this->base_prediction = nullptr;
}

PredictionDetection::PredictionDetection(std::shared_ptr<IPrediction>
    base) {
  this->base_prediction = base;
}

PredictionDetection::~PredictionDetection() {
  this->base_prediction = nullptr;
  this->bounding_boxes.clear();
}

RuntimeError PredictionDetection::SetBoundingBoxes(BBox *bounding_boxes,
    unsigned int size) {
  RuntimeError error;

  if (nullptr == bounding_boxes) {
    error.Set(RuntimeError::Code::NULL_PARAMETER, "Null bounding boxes pointer");
    return error;
  }
  if (0 == size) {
    error.Set(RuntimeError::Code::INCOMPATIBLE_PARAMETERS,
              "Bounding boxes array size is zero");
    return error;
  }

  this->bounding_boxes = std::vector<BBox>(bounding_boxes,
                         bounding_boxes + (size * sizeof(BBox)) );
  return error;
}

BBox *PredictionDetection::GetBoundingBoxes() {
  return this->bounding_boxes.data();
}

}