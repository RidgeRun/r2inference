/*
 * Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include <vector>

#include <r2i/preprocessing/normalize_resnet50v1.h>

/* Architecture specific required dimensions Resnet50V1 */
#define REQ_WIDTH_224 224
#define REQ_HEIGTH_224 224

/* Constants for preprocessing */
#define MEAN_RED 123.68
#define MEAN_GREEN 116.78
#define MEAN_BLUE 103.94
#define STD_DEV 1

namespace r2i {

NormalizeResnet50V1::NormalizeResnet50V1 () : r2i::Normalize() {
  /* Set supported dimensions for Resnet50V1 architecture */
  this->dimensions.push_back(std::tuple<int, int>(REQ_WIDTH_224, REQ_HEIGTH_224));
}

r2i::RuntimeError NormalizeResnet50V1::SetNormalizationParameters (
  unsigned char *frame_data, int width, int height,
  int channels) {
  this->mean_red = MEAN_RED;
  this->mean_green = MEAN_GREEN;
  this->mean_blue = MEAN_BLUE;
  this->std_dev_red = STD_DEV;
  this->std_dev_green = STD_DEV;
  this->std_dev_blue = STD_DEV;
  return r2i::RuntimeError();
}

}  // namespace r2i

r2i::IPreprocessing *
FactoryMakePreprocessing () {
  return new r2i::NormalizeResnet50V1 ();
}
