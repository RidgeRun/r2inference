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
#include <r2i/runtimeerror.h>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace r2i {
namespace onnxrt {

class Model : public IModel {
 public:
  Model();

  std::shared_ptr<Ort::Session> GetOnnxrtSession();

  RuntimeError Start(const std::string &name) override;

  RuntimeError Set(std::shared_ptr<Ort::Session> onnxrt_session);
  std::vector<const char *> GetInputNodeNames(RuntimeError &error);
  std::vector<const char *> GetOutputNodeNames(RuntimeError &error);

 private:
  std::shared_ptr<Ort::Session> session_ptr;
  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
};
}  // namespace onnxrt
}  // namespace r2i

#endif  // R2I_ONNX_MODEL_H
