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

#include <r2i/loader.h>

#include <core/session/onnxruntime_cxx_api.h>

#include <memory>
#include <string>

#include <r2i/imodel.h>
#include <r2i/onnxrt/model.h>

namespace r2i {
namespace onnxrt {

class Loader : public r2i::Loader {
 public:
  virtual std::shared_ptr<r2i::IModel> Load(const std::string &in_path,
      r2i::RuntimeError &error) override;

 private:
  std::shared_ptr<Model> model;
  std::shared_ptr<char> model_data;
};
}  // namespace onnxrt
}  // namespace r2i

#endif  // R2I_ONNX_LOADER_H
