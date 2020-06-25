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

#ifndef R2I_ONNX_MODEL_H
#define R2I_ONNX_MODEL_H

#include <r2i/imodel.h>

#include <core/session/onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include <r2i/runtimeerror.h>

namespace r2i {
namespace onnxrt {

class Model : public IModel {
 public:
  Model();

  std::shared_ptr<Ort::Session> GetOnnxrtSession();
  RuntimeError Start(const std::string &name) override;
  RuntimeError SetOnnxrtModel(std::shared_ptr<void> model_data,
                              size_t model_data_size);
  std::shared_ptr<void> GetOnnxrtModel();
  size_t GetOnnxrtModelSize();

 private:
  std::shared_ptr<void> model_data;
  size_t model_data_size;
};
}  // namespace onnxrt
}  // namespace r2i

#endif  // R2I_ONNX_MODEL_H
