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

#include "r2i/ncsdk/frame.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

RuntimeError Frame::Configure (void *in_data, int width,
                               int height, r2i::ImageFormat::Id format) {
  RuntimeError error;
  ImageFormat imageformat (format);

  if (nullptr == in_data) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received a NULL data pointer");
    return error;
  }
  if (width <= 0) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Received an invalid image width");
    return error;
  }
  if (height <= 0) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Received an invalid image height");
    return error;
  }

  this->frame_data = static_cast<float *>(in_data);
  this->frame_width = width;
  this->frame_height = height;
  this->frame_format = imageformat;

  return error;
}

void *Frame::GetData () {
  return this->frame_data;
}

int Frame::GetWidth () {
  return this->frame_width;
}

int Frame::GetHeight () {
  return this->frame_height;
}

ImageFormat Frame::GetFormat () {
  return this->frame_format;
}

DataType Frame::GetDataType () {
  return r2i::DataType::Id::FLOAT;
}

}
}
