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

#include "detection.h"

namespace r2i {

Detection::Detection() {
  this->type = InferenceOutputType::OBJECT_DETECTION;
}

Detection::~Detection() {
  this->detections.clear();
}

RuntimeError Detection::SetDetections(std::vector< DetectionInstance >
                                      detections) {
  this->detections = detections;
  return RuntimeError();
}

std::vector< DetectionInstance > Detection::GetDetections() {
  return this->detections;
}

} // namespace r2
