/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include "r2i/tensorrt/frame.h"

#include <cuda.h>
#include <cuda_runtime.h>

namespace r2i {
namespace tensorrt {

Frame::Frame () :
  frame_data(nullptr), frame_width(0), frame_height(0),
  frame_format(ImageFormat::Id::UNKNOWN_FORMAT),
  data_type(DataType::Id::UNKNOWN_DATATYPE) {
}

RuntimeError Frame::Configure (void *in_data, int width,
                               int height, r2i::ImageFormat::Id format,
                               r2i::DataType::Id datatype_id) {
  RuntimeError error;
  cudaError_t cuda_error;
  ImageFormat image_format (format);

  DataType data_type (r2i::DataType::FLOAT);

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
  if (datatype_id == DataType::Id::UNKNOWN_DATATYPE) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Can not set Frame with unknown data type");
    return error;
  }
  if (format == ImageFormat::Id::UNKNOWN_FORMAT) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Can not set Frame with unknown frame format");
    return error;
  }

  /* FIXME cudaMalloc is an expensive operation, this should be only done when necessary */
  size_t frame_size = width * height * image_format.GetNumPlanes() *
                      data_type.GetBytesPerPixel();
  void *buff;
  cuda_error = cudaMalloc (&buff, frame_size);
  if (cudaSuccess != cuda_error) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Unable to allocate managed buffer");
    return error;
  }
  this->frame_data = std::shared_ptr <void> (buff, cudaFree);

  cuda_error = cudaMemcpy(this->frame_data.get(), in_data, frame_size,
                          cudaMemcpyHostToDevice);
  if (cudaSuccess != cuda_error) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Unable to copy data to buffer");
    return error;
  }

  this->frame_width = width;
  this->frame_height = height;
  this->frame_format = image_format;
  this->data_type = data_type;

  return error;
}

void *Frame::GetData () {
  return this->frame_data.get();
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
  return this->data_type;
}

}
}
