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

namespace r2i {
namespace tflite {

Prediction::Prediction () {
}

double Prediction::At (unsigned int index,  r2i::RuntimeError &error) {
  return 0.0;
}

void *Prediction::GetResultData () {
  return nullptr;
}

unsigned int Prediction::GetResultSize () {
  return 0;
}
}
}
