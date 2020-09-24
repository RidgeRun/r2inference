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

#include <memory>
#include <vector>

#include <r2i/preprocessing/normalize.h>

namespace r2i {

Normalize::Normalize () {
  /* Set supported formats */
  this->formats.push_back(r2i::ImageFormat(r2i::ImageFormat::Id::RGB));
}

r2i::RuntimeError Normalize::Apply(std::shared_ptr<r2i::IFrame> in_frame,
                                   std::shared_ptr<r2i::IFrame> out_frame) {
  r2i::RuntimeError error;
  int width = 0;
  int height = 0;
  int channels = 0;
  int required_width = 0;
  int required_height = 0;
  int required_channels = 0;
  r2i::ImageFormat format;
  r2i::ImageFormat required_format;
  r2i::ImageFormat::Id required_format_id;
  unsigned char *in_data;
  float *out_data;

  if (!in_frame or !out_frame) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER, "Null IFrame parameters");
    return error;
  }

  in_data = static_cast<unsigned char *>(in_frame->GetData());
  if (!in_data) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER, "Null input frame data");
    return error;
  }

  width = in_frame->GetWidth();
  height = in_frame->GetHeight();
  format = in_frame->GetFormat();
  channels = format.GetNumPlanes();

  out_data = static_cast<float *>(out_frame->GetData());
  if (!out_data) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER, "Null output frame data");
    return error;
  }

  required_width = out_frame->GetWidth();
  required_height = out_frame->GetHeight();
  required_format = out_frame->GetFormat();
  required_format_id = required_format.GetId();
  required_channels = required_format.GetNumPlanes();

  error = Validate(required_width, required_height, required_format_id);
  if (error.IsError ()) {
    return error;
  }

  PreProcessImage(in_data, out_data, width, height, channels, required_width,
                  required_height, required_channels, error);

  if (error.IsError ()) {
    return error;
  }
  return error;
}

std::vector<r2i::ImageFormat> Normalize::GetAvailableFormats() {
  return this->formats;
}

std::vector<std::tuple<int, int>>
Normalize::GetAvailableDataSizes() {
  return this->dimensions;
}

r2i::RuntimeError Normalize::Validate (int required_width,
                                       int required_height,
                                       r2i::ImageFormat::Id required_format_id) {

  r2i::RuntimeError error;
  r2i::ImageFormat format;
  bool match_dimensions = false;
  bool match_format = false;
  int width = 0;
  int height = 0;

  /* Verify if the required dimensions are supported */
  for (unsigned int i = 0; i < this->dimensions.size(); i++) {
    width = std::get<0>(this->dimensions.at(i));
    height = std::get<1>(this->dimensions.at(i));
    if (width == required_width and height == required_height) {
      match_dimensions = true;
      break;
    }
  }

  if (!match_dimensions) {
    error.Set (r2i::RuntimeError::Code::MODULE_ERROR,
               "Required output dimensions are not supported in the preprocessing module");
    return error;
  }

  /* Verify if the required format is supported */
  for (unsigned int i = 0; i < this->formats.size(); i++) {
    format = this->formats.at(i);
    if (format.GetId() == required_format_id) {
      match_format = true;
      break;
    }
  }

  if (!match_format) {
    error.Set (r2i::RuntimeError::Code::MODULE_ERROR,
               "Required output image format is not supported in the preprocessing module");
    return error;
  }

  return error;
}

r2i::RuntimeError Normalize::SetNormalizationParameters (
  unsigned char *frame_data, int width, int height,
  int channels) {
  return r2i::RuntimeError();
}

void Normalize::PreProcessImage (unsigned char *in_data, float *out_data,
                                 int width, int height, int channels, int required_width,
                                 int required_height, int required_channels,
                                 r2i::RuntimeError error) {

  const int scaled_size = required_channels * required_width * required_height;

  /* To set model specific preprocessing paramaters */
  SetNormalizationParameters(in_data, required_width, required_height,
                             channels);

  for (int i = 0; i < scaled_size; i += channels) {
    /* RGB = (RGB - Mean)/StdDev */
    out_data[i + 0] = (static_cast<float>(in_data[i + 0]) - this->mean_red) /
                      this->std_dev_red;
    out_data[i + 1] = (static_cast<float>(in_data[i + 1]) - this->mean_green) /
                      this->std_dev_green;
    out_data[i + 2] = (static_cast<float>(in_data[i + 2]) - this->mean_blue) /
                      this->std_dev_blue;
  }
}

}  // namespace r2i
