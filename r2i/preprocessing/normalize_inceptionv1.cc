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

#include <r2i/preprocessing/normalize_inceptionv1.h>

/* Architecture specific required dimensions InceptionV1 and InceptionV2 */
#define REQ_WIDTH_224 224
#define REQ_HEIGTH_224 224

/* Constants for preprocessing */
#define MEAN_128 128.0
#define STD_DEV_128 128.0

namespace r2i {

NormalizeInceptionV1::NormalizeInceptionV1 () : r2i::Normalize() {
  /* Set supported dimensions for InceptionV1 and InceptionV2 architectures */
  this->dimensions.push_back(std::tuple<int, int>(REQ_WIDTH_224, REQ_HEIGTH_224));
}

r2i::RuntimeError NormalizeInceptionV1::SetNormalizationParameters (
  unsigned char *frame_data, int width, int height,
  int channels) {
  this->mean_red = MEAN_128;
  this->mean_green = MEAN_128;
  this->mean_blue = MEAN_128;
  this->std_dev_red = STD_DEV_128;
  this->std_dev_green = STD_DEV_128;
  this->std_dev_blue = STD_DEV_128;
  return r2i::RuntimeError();
}

}  // namespace r2i

r2i::IPreprocessing *
FactoryMakePreprocessing () {
  return new r2i::NormalizeInceptionV1 ();
}
