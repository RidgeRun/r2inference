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

#include <cmath>
#include <vector>

#include <r2i/preprocessing/normalize_facenetv1.h>

/* Architecture specific required dimensions FaceNetV1 */
#define REQ_WIDTH_160 160
#define REQ_HEIGTH_160 160

namespace r2i {

NormalizeFaceNetV1::NormalizeFaceNetV1 () : r2i::Normalize() {
  /* Set supported dimensions for TinyyoloV3 architecture */
  this->dimensions.push_back(std::tuple<int, int>(REQ_WIDTH_160, REQ_HEIGTH_160));
}

r2i::RuntimeError NormalizeFaceNetV1::SetNormalizationParameters (
  unsigned char *frame_data, int width, int height,
  int channels) {
  double mean = 0;
  double std_dev = 1;
  double sum = 0;
  double normalized = 0;
  double variance = 0;
  double red = 0;
  double green = 0;
  double blue = 0;
  int size = 0;
  unsigned char *data;
  r2i::RuntimeError error;

  data = frame_data;
  size = width * height * channels;

  /* Calculate mean */
  for (int i = 0; i < size; i += channels) {
    sum += data[i + 0];
    sum += data[i + 1];
    sum += data[i + 2];
  }
  mean = sum / (float) (size);

  /* Calculate std_dev */
  for (int i = 0; i < size; i += channels) {
    red = data[i + 0] - mean;
    green = data[i + 1] - mean;
    blue = data[i + 2] - mean;
    normalized += std::pow(red, 2);
    normalized += std::pow(green, 2);
    normalized += std::pow(blue, 2);
  }

  variance = normalized / (float) (size);
  std_dev = std::sqrt (variance);

  this->mean_red = mean;
  this->mean_green = mean;
  this->mean_blue = mean;
  this->std_dev_red = std_dev;
  this->std_dev_green = std_dev;
  this->std_dev_blue = std_dev;
  return error;
}

}  // namespace r2i

r2i::IPreprocessing *
FactoryMakePreprocessing () {
  return new r2i::NormalizeFaceNetV1 ();
}
