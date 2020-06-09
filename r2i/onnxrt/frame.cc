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

#include "r2i/onnxrt/frame.h"

#include <core/session/onnxruntime_cxx_api.h>

namespace r2i {
namespace onnxrt {

Frame::Frame () :
  frame_data(nullptr), frame_width(0), frame_height(0),
  frame_format(ImageFormat::Id::UNKNOWN_FORMAT) {
}

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

std::shared_ptr<Ort::Value> Frame::GetInputTensor (std::shared_ptr<Ort::Session>
    session, RuntimeError &error) {
  error.Clean ();

  int width = this->frame_width;
  int height = this->frame_height;
  int channels = this->frame_format.GetNumPlanes();
  size_t input_shape_size = 0;
  size_t input_image_size = width * height * channels;
  int64_t *input_shape;
  std::vector<int64_t> input_node_dims;

  if (!session) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Using null pointer for onnxruntime session");
    return nullptr;
  }

  // Warning: We support batches of size 1 and models with 1 input only
  if (session->GetInputCount() > 1) {
    error.Set(RuntimeError::Code::INCOMPATIBLE_MODEL,
              "Number of inputs in the model is greater than 1, this is not supported");
    return nullptr;
  }

  try {
    this->CreateTypeInfo(session, 0);
    this->CreateTensorTypeAndShapeInfo();
    input_node_dims = this->GetTensorShape();
    input_shape_size = input_node_dims.size();
  }

  catch (std::exception &excep) {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR, excep.what());
    return nullptr;
  }

  // Expected number of tensor dimensions is 4
  if (input_shape_size == 4) {
    input_shape = new int64_t[input_shape_size];
  }

  else {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR,
              "Number of input dimensions is different from 4");
    return nullptr;
  }

  for (unsigned int i = 0; i < input_node_dims.size(); i++) {
    input_shape [i] = input_node_dims[i];
  }

  // Input tensor and input image dimensions have to match
  error = this->ValidateTensorShape (input_shape);
  if (error.IsError ()) {
    delete []input_shape;
    return nullptr;
  }

  try {
    this->CreateMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault);
    this->CreateTensor(input_image_size, input_shape, input_shape_size);
  }

  catch (std::exception &excep) {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR, excep.what());
    delete []input_shape;
    return nullptr;
  }

  delete []input_shape;
  return this->input_tensor_ptr;
}

std::vector<int64_t> Frame::GetTensorShape() {
  std::vector<int64_t> shape_vector = this->tensor_info_ptr->GetShape();
  return shape_vector;
}

void Frame::CreateTypeInfo(std::shared_ptr<Ort::Session> session,
                           int input_index) {
  Ort::TypeInfo type_info = session->GetInputTypeInfo(input_index);
  this->type_info_ptr = std::shared_ptr<Ort::TypeInfo>(&type_info);
}

void Frame::CreateTensorTypeAndShapeInfo() {
  Ort::TensorTypeAndShapeInfo tensor_info =
    this->type_info_ptr->GetTensorTypeAndShapeInfo();
  this->tensor_info_ptr = std::shared_ptr<Ort::TensorTypeAndShapeInfo>
                          (&tensor_info);
}

void Frame::CreateMemoryInfo(OrtAllocatorType allocator_type,
                             OrtMemType mem_type) {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(allocator_type, mem_type);
  this->memory_info_ptr = std::shared_ptr<Ort::MemoryInfo>(&mem_info);
}

void Frame::CreateTensor(size_t element_count, int64_t *shape,
                         size_t shape_len) {
  Ort::Value value = Ort::Value::CreateTensor(
                       *this->memory_info_ptr, this->frame_data, element_count, shape,
                       shape_len);
  this->input_tensor_ptr = std::shared_ptr<Ort::Value>(&value);
}

RuntimeError Frame::ValidateTensorShape (int64_t *shape) {
  RuntimeError error;
  int frame_format_channels = this->frame_format.GetNumPlanes();

  // We only support 1 batch
  if (1 != shape[0]) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "We only support a batch of 1 image(s) in our frames");
    return error;
  }

  // Check that channels match
  if (frame_format_channels != shape[1]) {
    std::string error_msg;
    error_msg = "Channels per image:" + std::to_string(frame_format_channels) +
                ", needs to be equal to model input channels:" + std::to_string(shape[3]);
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, error_msg);
    return error;
  }

  // Check that heights match
  if (this->frame_height != shape[2]) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Unsupported image height");
    return error;
  }

  // Check that widths match
  if (this->frame_width != shape[3]) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Unsupported image width");
    return error;
  }

  return error;
}

}  // namespace onnxrt
}  // namespace r2i
