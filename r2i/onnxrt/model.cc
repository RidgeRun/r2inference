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

#include <core/common/exceptions.h>
#include <core/session/onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace r2i {
namespace onnxrt {

Model::Model() {
  this->session_ptr = nullptr;
}

RuntimeError Model::Start(const std::string &name) {
  RuntimeError error;

  return error;
}

RuntimeError Model::Set(std::shared_ptr<Ort::Session> onnxrt_session) {
  RuntimeError error;

  if (nullptr == onnxrt_session) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Trying to set model session with null session pointer");
    return error;
  }

  this->session_ptr = onnxrt_session;

  return error;
}

std::shared_ptr<Ort::Session> Model::GetOnnxrtSession() {
  return this->session_ptr;
}

}  // namespace onnxrt
}  // namespace r2i
