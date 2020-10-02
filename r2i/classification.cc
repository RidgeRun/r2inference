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

#include "classification.h"

namespace r2i {

Classification::Classification() {
  this->type = InferenceOutputType::CLASSIFICATION;
}

Classification::~Classification() {
  this->labels.clear();
}

RuntimeError Classification::SetLabels(std::vector< ClassificationInstance >
                                       labels) {
  this->labels = labels;
  return RuntimeError();
}

std::vector< ClassificationInstance > Classification::GetLabels() {
  return this->labels;
}

}
