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

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/string_util.h>

#include "r2i/tflite/frame.h"
#include "r2i/tflite/prediction.h"

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
    this->SetupResolver(resolver);

    ::tflite::ErrorReporter *error_reporter = ::tflite::DefaultErrorReporter();

    std::unique_ptr<::tflite::Interpreter> interpreter;

    ::tflite::InterpreterBuilder(this->model->GetTfliteModel()->GetModel(),
                                 resolver, error_reporter)(&interpreter);

    if (!interpreter) {
      error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
                 "Failed to construct interpreter");
      return error;
    }

    this->SetInterpreterContext(interpreter.get());

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

std::shared_ptr<r2i::IPrediction> Engine::Process (std::shared_ptr<r2i::IFrame>
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

  this->PreprocessInputData(static_cast<float *>(frame->GetData()),
                            wanted_width * wanted_height * wanted_channels, this->interpreter.get(), error);
  if (r2i::RuntimeError::EOK != error.GetCode()) {
    return nullptr;
  }

  if (this->interpreter->Invoke() != kTfLiteOk) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Failed to invoke tflite!");
    return nullptr;
  }

  std::vector<float> tensor_data;
  this->GetOutputTensorData(this->interpreter.get(), tensor_data, error);
  if (r2i::RuntimeError::EOK != error.GetCode()) {
    return nullptr;
  }

  prediction->SetTensorValues(tensor_data.data(), tensor_data.size());

  return prediction;
}

Engine::~Engine () {
  this->Stop();
}

template <typename T1, typename T2>
static T2 ConvertToFixedPoint(const T1 value, const float scale,
                              const int zero_point) {
  return static_cast<T2>((value / scale) + zero_point);
}

template <typename T1, typename T2>
static void ConvertArrayToFixedPoint(const T1 *data, T2 *output_data,
                                     const int size, const float scale, const int zero_point) {
  for (int index = 0; index < size; index++) {
    output_data[index] = static_cast<T2>(ConvertToFixedPoint<T1, T2>(data[index],
                                         scale, zero_point));
  }
}

template <typename T1, typename T2>
static T1 ConvertToFloatingPoint(const T2 value, const float scale,
                                 const int zero_point) {
  return static_cast<T1>((value - zero_point) * scale);
}

template <typename T1, typename T2>
static void ConvertArrayToFloatingPoint(const T2 *data,
                                        std::vector<T1> &output_data, const int size, const float scale,
                                        const int zero_point) {
  for (int index = 0; index < size; index++) {
    output_data[index] = ConvertToFloatingPoint<T1, T2>(data[index], scale,
                         zero_point);
  }
}

void Engine::SetupResolver(::tflite::ops::builtin::BuiltinOpResolver
                           &/*resolver*/) {
  // No implementation for tflite engine
}

void Engine::SetInterpreterContext(::tflite::Interpreter * /*interpreter*/) {
  // No implementation for tflite engine
}

void Engine::PreprocessInputData(const float *input_data, const int size,
                                 ::tflite::Interpreter *interpreter, r2i::RuntimeError &error) {
  const auto &input_indices = interpreter->inputs();
  const auto *tensor = interpreter->tensor(input_indices[0]);

  if (!input_data) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE, "Failed to get image data");
    return;
  }

  if (kTfLiteUInt8 == tensor->type) {
    auto input_fixed_tensor = interpreter->typed_input_tensor<uint8_t>(0);
    ConvertArrayToFixedPoint<float, uint8_t>(input_data, input_fixed_tensor, size,
        tensor->params.scale, tensor->params.zero_point);
  } else if (kTfLiteFloat32 == tensor->type) {
    auto input_tensor = interpreter->typed_tensor<float>(input_indices[0]);
    memcpy(input_tensor, input_data, size * sizeof(float));
  } else {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Output tensor has unsupported output type");
    return;
  }
}

void Engine::GetOutputTensorData(::tflite::Interpreter *interpreter,
                                 std::vector<float> &output_data,
                                 r2i::RuntimeError &error) {

  const auto &output_indices = interpreter->outputs();
  const auto *out_tensor = interpreter->tensor(output_indices[0]);

  if (nullptr == out_tensor) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Output tensor is null");
    return;
  }

  if (kTfLiteUInt8 == out_tensor->type) {
    const int num_values = out_tensor->bytes;
    output_data.resize(num_values);
    const uint8_t *output = interpreter->typed_output_tensor<uint8_t>(0);
    ConvertArrayToFloatingPoint<float, uint8_t>(output, output_data, num_values,
        out_tensor->params.scale, out_tensor->params.zero_point);
  } else if (kTfLiteFloat32 == out_tensor->type) {
    const int num_values = out_tensor->bytes / sizeof(float);
    output_data.resize(num_values);
    const float *output = interpreter->typed_output_tensor<float>(0);
    memcpy(&output_data[0], output, num_values * sizeof(float));
  } else {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Output tensor has unsupported output type");
    return;
  }
}

} //namespace tflite
} //namepsace r2i
