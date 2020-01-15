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

#include "r2i/tflite/frame.h"

namespace r2i {
namespace tflite {

Frame::Frame () :
  frame_data(nullptr), frame_width(0), frame_height(0),
  frame_format(ImageFormat::Id::UNKNOWN_FORMAT) {
}

RuntimeError Frame::Configure (void *in_data, int width,
                               int height, r2i::ImageFormat::Id format) {
  RuntimeError error;

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

}
}
