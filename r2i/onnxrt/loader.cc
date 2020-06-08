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

#include "r2i/onnxrt/loader.h"

#include <onnxruntime/core/common/exceptions.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <fstream>
#include <memory>
#include <string>

#include "r2i/imodel.h"
#include "r2i/onnxrt/model.h"

namespace r2i {
namespace onnxrt {

std::shared_ptr<r2i::IModel> Loader::Load(const std::string &in_path,
    r2i::RuntimeError &error) {
  if (in_path.empty()) {
    error.Set(RuntimeError::Code::WRONG_API_USAGE,
              "Received NULL path to file");
    return nullptr;
  }

  std::ifstream graphdef_file;
  graphdef_file.open(in_path, std::ios::binary | std::ios::ate);
  if (false == graphdef_file.is_open()) {
    error.Set(RuntimeError::Code::FILE_ERROR, "Unable to open file");
    return nullptr;
  }
  graphdef_file.close();

  try {
    this->CreateEnv(ORT_LOGGING_LEVEL_WARNING, in_path);
    this->CreateSessionOptions();
    this->CreateSession(this->env_ptr, in_path, this->session_options_ptr);
  }

  catch (std::exception &excep) {
    error.Set(RuntimeError::Code::FRAMEWORK_ERROR, excep.what());
  }

  if (error.IsError()) {
    return nullptr;
  }

  auto model = std::make_shared<Model>();

  error = model->Set(this->session_ptr);

  if (error.IsError()) {
    return nullptr;
  }

  return model;
}

void Loader::CreateEnv(OrtLoggingLevel log_level, const std::string &log_id) {
  this->env_ptr = std::make_shared<Ort::Env>(log_level, log_id.c_str());
}

void Loader::CreateSessionOptions() {
  // TODO: This options should be paramaters in the class. Add method
  // to pass this options inside the class.
  this->session_options_ptr = std::make_shared<Ort::SessionOptions>();
  this->session_options_ptr->SetIntraOpNumThreads(1);
  this->session_options_ptr->SetGraphOptimizationLevel(
    GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

void Loader::CreateSession(std::shared_ptr<Ort::Env> env,
                           const std::string &name,
                           std::shared_ptr<Ort::SessionOptions> options) {
  this->session_ptr =
    std::make_shared<Ort::Session>(*env, name.c_str(), *options);
}

}  // namespace onnxrt
}  // namespace r2i
