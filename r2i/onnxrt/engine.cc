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

#include "r2i/onnxrt/engine.h"

#include <core/common/exceptions.h>
#include <core/session/onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include "r2i/onnxrt/frame.h"
#include "r2i/onnxrt/model.h"
#include "r2i/onnxrt/prediction.h"

namespace r2i {
namespace onnxrt {

Engine::Engine () : state(State::STOPPED), model(nullptr) {
}

Engine::~Engine () {
  this->Stop();
}

RuntimeError Engine::SetModel (std::shared_ptr<r2i::IModel> in_model) {
  RuntimeError error;

  if (State::STOPPED != this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Stopped model before setting a new state");
    return error;
  }

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null model pointer");
    return error;
  }
  auto model = std::dynamic_pointer_cast<r2i::onnxrt::Model, r2i::IModel>
               (in_model);

  if (nullptr == model) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "The provided model is not an ONNXRT model");
    return error;
  }

  this->model = model;

  this->session = this->model->GetOnnxrtSession();

  return error;
}

RuntimeError Engine::Start ()  {
  RuntimeError error;

  if (State::STARTED == this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine already started");
    return error;
  }

  if (nullptr == this->model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Model not set yet");
    return error;
  }

  this->state = State::STARTED;

  return error;
}

RuntimeError Engine::Stop () {
  RuntimeError error;

  if (State::STOPPED == this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine already stopped");
  }

  this->state = State::STOPPED;

  return error;
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) {
  ImageFormat frame_format;
  int frame_width = 0;
  int frame_height = 0;
  int frame_channels = 0;
  std::vector<int64_t> input_node_dims;
  size_t input_image_size = 0;

  auto prediction = std::make_shared<Prediction>();

  error.Clean ();

  if (State::STARTED != this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine not started");
    return nullptr;
  }

  auto frame = std::dynamic_pointer_cast<Frame, IFrame> (in_frame);
  if (!frame) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "The provided frame is not an onnxrt frame");
    return nullptr;
  }

  if (!this->session) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Received null onnxrt session pointer, can not do prediction");
    return nullptr;
  }

  if (this->session->GetInputCount() > 1) {
    error.Set(RuntimeError::Code::INCOMPATIBLE_MODEL,
              "Number of inputs in the model is greater than 1, this is not supported");
    return nullptr;
  }

  if (this->session->GetOutputCount() > 1) {
    error.Set(RuntimeError::Code::INCOMPATIBLE_MODEL,
              "Number of outputs in the model is greater than 1, this is not supported");
    return nullptr;
  }

  // Warning: We support batches of size 1 and models with
  // 1 input only.
  try {
    input_node_dims = this->session->GetInputTypeInfo(
                        0).GetTensorTypeAndShapeInfo().GetShape();
  }

  catch (std::exception &excep) {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR, excep.what());
    return nullptr;
  }

  frame_format = frame->GetFormat();
  frame_channels = frame_format.GetNumPlanes();
  frame_height = frame->GetHeight();
  frame_width = frame->GetWidth();
  input_image_size = frame_height * frame_width * frame_channels;

  error = this->ValidateInputTensorShape(frame_channels, frame_height,
                                         frame_width, input_node_dims);
  if (error.IsError ()) {
    return nullptr;
  }

  // Score model with input tensor, get back Prediction set with pointer
  // of the output tensor result.
  // Note that this implementation only supports 1 input and 1 output models.
  error = this->ScoreModel(this->session, frame, input_image_size,
                           input_node_dims,
                           prediction);

  if (error.IsError ()) {
    return nullptr;
  }

  return prediction;
}

RuntimeError Engine::ValidateInputTensorShape (int channels, int height,
    int width, std::vector<int64_t> input_dims) {
  RuntimeError error;

  // We only support 1 batch
  if (1 != input_dims.at(0)) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "We only support a batch of 1 image(s) in our frames");
    return error;
  }

  // Check that channels match
  if (channels != input_dims.at(1)) {
    std::string error_msg;
    error_msg = "Channels per image:" + std::to_string(channels) +
                ", needs to be equal to model input channels:" +
                std::to_string(input_dims.at(1));
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, error_msg);
    return error;
  }

  // Check that heights match
  if (height != input_dims.at(2)) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Unsupported image height");
    return error;
  }

  // Check that widths match
  if (width != input_dims.at(3)) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Unsupported image width");
    return error;
  }

  return error;
}

RuntimeError Engine::ScoreModel (std::shared_ptr<Ort::Session> session,
                                 std::shared_ptr<Frame> frame,
                                 size_t input_image_size,
                                 std::vector<int64_t> input_node_dims,
                                 std::shared_ptr<Prediction> prediction) {
  RuntimeError error;
  float *result;
  int64_t output_size;
  size_t num_input_nodes = this->session->GetInputCount();
  size_t num_output_nodes = this->session->GetOutputCount();
  std::vector<int64_t> output_node_dims;
  Ort::Value input_tensor{nullptr};
  Ort::AllocatorWithDefaultOptions input_allocator;
  Ort::AllocatorWithDefaultOptions output_allocator;

  // Warning: We support batches of size 1 and models with
  // 1 output only.
  try {
    output_node_dims = this->session->GetOutputTypeInfo(
                         0).GetTensorTypeAndShapeInfo().GetShape();
    output_size = this->session->GetOutputTypeInfo(
                    0).GetTensorTypeAndShapeInfo().GetElementCount();
  }

  catch (std::exception &excep) {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR, excep.what());
    return error;
  }

  num_input_nodes = this->session->GetInputCount();
  num_output_nodes = this->session->GetOutputCount();

  std::vector<const char *> input_node_names(num_input_nodes);
  std::vector<const char *> output_node_names(num_output_nodes);

  // Warning: only 1 input and output supported
  char *input_name = this->session->GetInputName(0, input_allocator);
  input_node_names[0] = input_name;
  char *output_name = this->session->GetOutputName(0, output_allocator);
  output_node_names[0] = output_name;

  try {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                  OrtMemTypeDefault);
    input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                   reinterpret_cast<float *>(frame->GetData()),
                   input_image_size,
                   input_node_dims.data(),
                   input_node_dims.size());
    auto output_tensor =
      this->session->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                         &input_tensor, num_input_nodes, output_node_names.data(),
                         num_output_nodes);

    result = output_tensor.at(0).GetTensorMutableData<float>();
  }

  catch (std::exception &excep) {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR, excep.what());
    return error;
  }

  prediction->SetTensorValues(result, output_size);

  return error;
}

}  // namespace onnxrt
}  // namespace r2i
