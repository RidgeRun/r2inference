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

#include "r2i/tflite/engine.h"

#include "r2i/tflite/prediction.h"
#include "r2i/tflite/frame.h"
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/string_util.h>

namespace r2i {
namespace tflite {

Engine::Engine () : state(State::STOPPED), model(nullptr) {
  this->number_of_threads = 0;
  this->allow_fp16 = 0;
}

RuntimeError Engine::SetModel (std::shared_ptr<r2i::IModel> in_model) {

  RuntimeError error;

  if (State::STOPPED != this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Stop model before setting a new state");
    return error;
  }

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null model");
    return error;
  }
  auto model = std::dynamic_pointer_cast<r2i::tflite::Model, r2i::IModel>
               (in_model);

  if (nullptr == model) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "The provided model is not an TFLITE model");
    return error;
  }

  if (nullptr != this->model) {
    this->model = nullptr;
  }

  this->model = model;

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

  if (!this->interpreter) {
    ::tflite::ops::builtin::BuiltinOpResolver resolver;
    this->setupResolver(resolver);

    ::tflite::ErrorReporter *error_reporter = ::tflite::DefaultErrorReporter();

    std::unique_ptr<::tflite::Interpreter> interpreter;

    ::tflite::InterpreterBuilder(this->model->GetTfliteModel()->GetModel(),
                                 resolver, error_reporter)(&interpreter);

    if (!interpreter) {
      error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
                 "Failed to construct interpreter");
      return error;
    }

    this->setInterpreterContext();

    std::shared_ptr<::tflite::Interpreter> tflite_interpreter_shared{std::move(interpreter)};

    this->interpreter = tflite_interpreter_shared;
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

RuntimeError Engine::SetNumberOfThreads (int number_of_threads) {
  RuntimeError error;

  /* Check if number of threads is greater than 0 */
  if (number_of_threads < 0 ) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "The number of threads needs to be greater than 0");
    return error;
  }
  this->number_of_threads = number_of_threads;
  return error;
}

const int Engine::GetNumberOfThreads () {
  return this->number_of_threads;
}

RuntimeError Engine::SetAllowFP16 (int allow_fp16) {
  this->allow_fp16 = allow_fp16;
  return RuntimeError ();
}
const int Engine::GetAllowFP16 () {
  return this->allow_fp16;
}

int64_t Engine::GetRequiredBufferSize (TfLiteIntArray *dims) {
  int64_t size = 1;

  /* For each dimension, multiply the amount of entries */
  for (int dim = 0; dim < dims->size; ++dim) {
    size *= dims->data[dim];
  }

  return size;
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) {
  ImageFormat in_format;

  error.Clean ();

  if (State::STARTED != this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine not started");
    return nullptr;
  }

  auto frame = std::dynamic_pointer_cast<Frame, IFrame> (in_frame);
  if (nullptr == frame) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "The provided frame is not an tensorflow lite frame");
    return nullptr;
  }

  if (this->number_of_threads > 0) {
    interpreter->SetNumThreads(this->number_of_threads);
  }

  interpreter->SetAllowFp16PrecisionForFp32(this->allow_fp16);

  if (this->interpreter->AllocateTensors() != kTfLiteOk) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Failed to allocate tensors!");
    return nullptr;
  }

  auto prediction = std::make_shared<Prediction>();

  int input = this->interpreter->inputs()[0];
  TfLiteIntArray *dims = this->interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  if ((frame->GetWidth() != wanted_width)
      or (frame->GetHeight() != wanted_height)) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "The provided frame input sizes are different to tensor sizes");
    return nullptr;
  }

  // Check the model quantization, only 32 bits allowed
  if (kTfLiteFloat32 != interpreter->tensor(input)->type) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "The provided model quantization is not allowed, only float32 is supported");
    return nullptr;
  }

  auto tensor_data = this->runInference(frame, input, wanted_width, wanted_height,
                                        wanted_channels, error);
  if (r2i::RuntimeError::EOK != error.GetCode()) {
    return nullptr;
  }

  int output = this->interpreter->outputs()[0];
  TfLiteIntArray *output_dims = this->interpreter->tensor(output)->dims;
  auto output_size = GetRequiredBufferSize(output_dims) * sizeof(float);
  prediction->SetTensorValues(tensor_data, output_size);

  return prediction;
}

Engine::~Engine () {
  this->Stop();
}

void Engine::setupResolver(::tflite::ops::builtin::BuiltinOpResolver
                           &/*resolver*/) {
  // No implementation for tflite engine
}

void Engine::setInterpreterContext() {
  // No implementation for tflite engine
}

float *Engine::runInference(std::shared_ptr<r2i::IFrame> frame,
                            const int &input, const int &width, const int &height, const int &channels,
                            r2i::RuntimeError &error) {
  auto input_tensor = this->interpreter->typed_tensor<float>(input);
  auto input_data = (float *)frame->GetData();

  if (!input_data) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, "Failed to get image data");
    return nullptr;
  }

  memcpy(input_tensor, input_data,
         height * width * channels * sizeof(float));

  if (this->interpreter->Invoke() != kTfLiteOk) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Failed to invoke tflite!");
    return nullptr;
  }

  return this->interpreter->typed_output_tensor<float>(0);
}

} //namespace tflite
} //namepsace r2i
