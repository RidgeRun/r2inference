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

#include <r2i/preprocessing/normalize_tinyyolov2.h>

/* Architecture specific required dimensions TinyyoloV2 */
#define REQ_WIDTH_416 416
#define REQ_HEIGTH_416 416

/* Constants for preprocessing */
#define MEAN 0
#define STD_DEV 255

namespace r2i {

NormalizeTinyyoloV2::NormalizeTinyyoloV2 () : r2i::Normalize() {
  /* Set supported dimensions for TinyyoloV2 architecture */
  this->dimensions.push_back(std::tuple<int, int>(REQ_WIDTH_416, REQ_HEIGTH_416));
}

r2i::RuntimeError NormalizeTinyyoloV2::SetNormalizationParameters (
  unsigned char *frame_data, int width, int height,
  int channels) {
  this->mean_red = MEAN;
  this->mean_green = MEAN;
  this->mean_blue = MEAN;
  this->std_dev_red = STD_DEV;
  this->std_dev_green = STD_DEV;
  this->std_dev_blue = STD_DEV;
  return r2i::RuntimeError();
}

}  // namespace r2i

r2i::IPreprocessing *
FactoryMakePreprocessing () {
  return new r2i::NormalizeTinyyoloV2 ();
}
