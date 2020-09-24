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

#include "r2i/tensorflow/frame.h"

namespace r2i {
namespace tensorflow {

Frame::Frame () :
  frame_data(nullptr), frame_width(0), frame_height(0),
  frame_format(ImageFormat::Id::UNKNOWN_FORMAT), tensor(nullptr) {
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

std::shared_ptr<TF_Tensor> Frame::GetTensor (std::shared_ptr<TF_Graph> graph,
    TF_Operation *operation, RuntimeError &error) {
  error.Clean ();

  if (nullptr != this->tensor) {
    return this->tensor;
  }

  TF_DataType type;
  int64_t *dims;
  int64_t num_dims;
  int64_t size;
  error = this->GetTensorShape (graph, operation, type, &dims, num_dims, size);
  if (error.IsError ()) {
    return nullptr;
  }

  error = this->Validate (dims, num_dims);
  if (error.IsError ()) {
    delete []dims;
    return nullptr;
  }

  error = this->CreateTensor (type, dims, num_dims, size);
  if (error.IsError ()) {
    delete []dims;
    return nullptr;
  }

  delete []dims;

  return this->tensor;
}

RuntimeError Frame::Validate (int64_t dims[], int64_t num_dims) {
  RuntimeError error;
  int frame_format_channels = this->frame_format.GetNumPlanes();

  /* We only support 1 batch */
  if (1 != dims[0]) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "We only support a batch of 1 image(s) in our frames");
    return error;
  }

  /* Check that widths match */
  if (this->frame_width != dims[1]) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Unsupported image width");
    return error;
  }

  /* Check that heights match */
  if (this->frame_height != dims[2]) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Unsupported image height");
    return error;
  }

  /* Check that channels match */
  if (frame_format_channels != dims[3]) {
    std::string error_msg;
    error_msg = "Channels per image:" + std::to_string(frame_format_channels) +
                ", needs to be equal to model input channels:" + std::to_string(dims[3]);
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, error_msg);
    return error;
  }

  return error;
}

static void DummyDeallocator (void *data, size_t len, void *arg) {
  //NOP
}

RuntimeError Frame::CreateTensor (TF_DataType type, int64_t dims[],
                                  int64_t num_dims, int64_t size) {
  RuntimeError error;

  TF_Tensor *raw_tensor = TF_NewTensor(type, dims, num_dims, this->frame_data,
                                       size, DummyDeallocator, NULL);
  if (nullptr == raw_tensor) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Unable to create input tensor");
    return error;
  }

  std::shared_ptr<TF_Tensor> tensor (raw_tensor, TF_DeleteTensor);
  this->tensor = tensor;

  return error;
}

void Frame::HandleGenericDimensions (int64_t dims[], int64_t num_dims) {
  const int64_t expected_dims = 4;
  /* TensorFlow assigns a generic (-1) value on dimensions that accept any
   * value. This is almost always true for the batch (first dimension), but
   * convolutional neural networks may also accept any width, height or even
   * amount of channels.
   */

  /* The idea above is only true if we have 4 dimensions, aka: BxWxHxC */
  if (expected_dims != num_dims) {
    return;
  }

  /* Is batch size generic? */
  if (-1 == dims[0]) {
    dims[0] = 1;
  }

  /* Is width generic? */
  if (-1 == dims[1]) {
    dims[1] = this->frame_width;;
  }

  /* Is height generic? */
  if (-1 == dims[2]) {
    dims[2] = this->frame_height;;
  }

  /* Is channels generic? */
  if (-1 == dims[3]) {
    dims[3] = this->frame_format.GetNumPlanes();
  }
}

RuntimeError Frame::GetTensorShape (std::shared_ptr<TF_Graph> pgraph,
                                    TF_Operation *operation, TF_DataType &type, int64_t **dims,
                                    int64_t &num_dims, int64_t &size) {
  RuntimeError error;
  TF_Graph *graph = pgraph.get();

  if (nullptr == graph) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Attempting to validate frame with NULL graph");
    return error;
  }

  if (nullptr == operation) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Attempting to validate frame with NULL operation");
    return error;
  }

  TF_Output output = {.oper = operation, .index = 0};
  std::shared_ptr<TF_Status> pstatus (TF_NewStatus (), TF_DeleteStatus);
  TF_Status *status = pstatus.get ();

  num_dims = TF_GraphGetTensorNumDims (graph, output, status);
  if (TF_GetCode (status) != TF_OK) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
    return error;
  }

  *dims = new int64_t[num_dims];
  TF_GraphGetTensorShape(graph, output, *dims, num_dims, status);
  if (TF_GetCode (status) != TF_OK) {
    delete []dims;
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
    return error;
  }

  HandleGenericDimensions(*dims, num_dims);

  type = TF_OperationOutputType(output);
  size = TF_DataTypeSize(type);

  /* Get the required amount of space on the buffer, considering all
     dimensions */
  for (int d = 0; d < num_dims; ++d) {
    size *= (*dims)[d];
  }

  return error;
}

DataType Frame::GetDataType () {
  return r2i::DataType::Id::FLOAT;
}

}
}
