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

#include "r2i/onnxrt/model.h"

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/common/exceptions.h>

#include <iostream>
#include <string>

namespace r2i {
namespace onnxrt {

Model::Model() { this->session_ptr = nullptr; }

RuntimeError Model::Start(const std::string &name) {
  RuntimeError error;
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "r2i");
    // TODO: This options should be paramaters in the class. Add method
    // to pass this options inside the class.
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env, name.c_str(), session_options);
    this->session_ptr = &session;
  }

  catch (const Ort::Exception &OrtException) {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR,
              "Failed creating ORT session");
  }

  catch (const onnxruntime::OnnxRuntimeException &OnnxRuntimeException) {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR, "Incompatible ONNX model");
  }

  return error;
}

Ort::Session *Model::GetSession() { return this->session_ptr; }

}  // namespace onnxrt
}  // namespace r2i
