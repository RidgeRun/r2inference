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

#ifndef R2I_ONNX_LOADER_H
#define R2I_ONNX_LOADER_H

#include <r2i/iloader.h>

#include <core/session/onnxruntime_cxx_api.h>

#include <memory>
#include <string>

#include <r2i/imodel.h>
#include <r2i/onnxrt/model.h>

namespace r2i {
namespace onnxrt {

class Loader : public ILoader {
 public:
  virtual std::shared_ptr<r2i::IModel> Load(const std::string &in_path,
      r2i::RuntimeError &error) override;

 private:
  std::shared_ptr<Model> model;
  std::shared_ptr<Ort::Env> env_ptr;
  std::shared_ptr<Ort::SessionOptions> session_options_ptr;
  std::shared_ptr<Ort::Session> session_ptr;

  void CreateEnv(OrtLoggingLevel log_level, const std::string &log_id);
  void CreateSessionOptions();
  void CreateSession(std::shared_ptr<Ort::Env> env, const std::string &name,
                     std::shared_ptr<Ort::SessionOptions> options);
};
}  // namespace onnxrt
}  // namespace r2i

#endif  // R2I_ONNX_LOADER_H
