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

#include "r2i/onnxrt_openvino/engine.h"

#include <core/providers/openvino/openvino_provider_factory.h>

static const OrtApi *g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

namespace r2i {
namespace onnxrt_openvino {

Engine::Engine () : onnxrt::Engine() {
  hardware_option = "CPU_FP32";
}

void Engine::AppendSessionOptionsExecutionProvider(Ort::SessionOptions
    &session_options, r2i::RuntimeError &error) {
  OrtStatus *status = NULL;
  error.Clean ();

  status = OrtSessionOptionsAppendExecutionProvider_OpenVINO(session_options,
           this->hardware_option.c_str());
  if (status != NULL) {
    std::string status_msg = std::string(g_ort->GetErrorMessage(status));
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, status_msg);
    g_ort->ReleaseStatus(status);
  }
}

RuntimeError Engine::SetHardwareId (const std::string &hardware_id) {
  RuntimeError error;

  if (State::STARTED == this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Parameter can't be set, engine already started");
    return error;
  }

  this->hardware_option = hardware_id;

  return error;
}

const std::string Engine::GetHardwareId () {
  return this->hardware_option;
}

Engine::~Engine() {

}

} //namespace onnxrt_openvino
} //namespace r2i
