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

#include "parameters.h"
#include "parameteraccessors.h"

#include <string>

namespace r2i {
namespace onnxrt_openvino {

Parameters::Parameters () : r2i::onnxrt::Parameters() {
  ParamDesc hardware_id_desc = {
    {hardware_id_meta},
    std::make_shared<r2i::onnxrt_openvino::HardwareIdAccessor>()
  };
  parameter_map.emplace(std::make_pair(hardware_id_meta.name,
                                       hardware_id_desc));
}

RuntimeError Parameters::Configure (std::shared_ptr<r2i::IEngine> in_engine,
                                    std::shared_ptr<r2i::IModel> in_model) {
  RuntimeError error;

  if (nullptr == in_engine) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received null engine");
    return error;
  }

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received null model");
    return error;
  }

  auto engine = std::dynamic_pointer_cast<r2i::onnxrt_openvino::Engine, IEngine>
                (in_engine);
  if (nullptr == engine) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_ENGINE,
               "The provided engine is not an ONNXRT OpenVINO engine");
    return error;
  }

  auto model = std::dynamic_pointer_cast<r2i::onnxrt::Model, IModel>(in_model);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an ONNXRT model");
    return error;
  }

  this->engine = engine;
  this->model = model;

  return error;
}

RuntimeError Parameters::SetHardwareId (std::string &value) {
  RuntimeError error;
  error = this->engine->SetHardwareId(value);
  return error;
}

RuntimeError Parameters::GetHardwareId (std::string &value) {
  RuntimeError error;
  value = this->engine->GetHardwareId();
  return error;
}

}  // namespace onnxrt_openvino
}  // namespace r2i
