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

#include "r2i/edgetpu/engine.h"

#include <edgetpu.h>

namespace r2i {
namespace edgetpu {

Engine::Engine () {
  this->state = r2i::tflite::Engine::State::STOPPED;
  this->model = nullptr;
  this->number_of_threads = 1;
  this->allow_fp16 = 0;
}

void Engine::SetupResolver(::tflite::ops::builtin::BuiltinOpResolver
                           &resolver) {
  resolver.AddCustom(::edgetpu::kCustomOp, ::edgetpu::RegisterCustomOp());
}

void Engine::SetInterpreterContext() {
  std::shared_ptr<::edgetpu::EdgeTpuContext> edgetpu_context =
    ::edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  this->interpreter->SetExternalContext(kTfLiteEdgeTpuContext,
                                        edgetpu_context.get());
}

float *Engine::RunInference(std::shared_ptr<r2i::IFrame> frame,
                            const int &input, const int size,
                            r2i::RuntimeError &error) {
  std::unique_ptr<float> output_data;

  auto input_tensor = this->interpreter->typed_tensor<uint8_t>(input);
  auto input_data = static_cast<float *>(frame->GetData());

  if (!input_data) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE, "Failed to get image data");
    return nullptr;
  }

  // Convert to fixed point
  std::unique_ptr<uint8_t> input_data_fixed(new uint8_t(size));
  for (int index = 0; index < size; index++) {
    input_data_fixed.get()[index] = static_cast<uint8_t>(input_data[index]);
  }

  memcpy(input_tensor, input_data_fixed.get(),
         size * sizeof(uint8_t));

  if (this->interpreter->Invoke() != kTfLiteOk) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Failed to invoke tflite!");
    return nullptr;
  }

  const auto &output_indices = interpreter->outputs();
  const auto *out_tensor = interpreter->tensor(output_indices[0]);

  if (out_tensor->type == kTfLiteUInt8) {
    uint8_t *output_data_fixed = interpreter->typed_output_tensor<uint8_t>(0);
    TfLiteIntArray *output_dims = this->interpreter->tensor(
                                    output_indices[0])->dims;

    // Convert to fixed point
    auto output_size = GetRequiredBufferSize(output_dims);
    output_data = std::unique_ptr<float>(new float(output_size));
    for (int index = 0; index < output_size; index++) {
      output_data.get()[index] = static_cast<float>(output_data_fixed[index]);
    }
  } else if (out_tensor->type == kTfLiteFloat32) {
    output_data = std::unique_ptr<float>(interpreter->typed_output_tensor<float>
                                         (0));
  } else {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Output tensor has unsupported output type");
    return nullptr;
  }

  return output_data.get();
}

} //namepsace edgetpu
} //namepsace r2i
