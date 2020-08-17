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

#include <string>

namespace r2i {
namespace onnxrt_openvino {

Parameters::Parameters () {
  ParamDesc logging_level_desc = {
    {logging_level_meta},
    std::make_shared<LoggingLevelAccessor>(this)
  };
  parameter_map.emplace(std::make_pair(logging_level_meta.name,
                                       logging_level_desc));

  ParamDesc log_id_desc = {
    {log_id_meta},
    std::make_shared<LogIdAccessor>(this)
  };
  parameter_map.emplace(std::make_pair(log_id_meta.name, log_id_desc));

  ParamDesc intra_num_threads_desc = {
    {intra_num_threads_meta},
    std::make_shared<IntraNumThreadsAccessor>(this)
  };
  parameter_map.emplace(std::make_pair(intra_num_threads_meta.name,
                                       intra_num_threads_desc));

  ParamDesc graph_optimization_level_desc = {
    {graph_optimization_level_meta},
    std::make_shared<GraphOptLevelAccessor>(this)
  };
  parameter_map.emplace(std::make_pair(graph_optimization_level_meta.name,
                                       graph_optimization_level_desc));

  ParamDesc hardware_id_desc = {
    {hardware_id_meta},
    std::make_shared<HardwareIdAccessor>(this)
  };
  parameter_map.emplace(std::make_pair(hardware_id_meta.name, hardware_id_desc));
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

  auto engine = std::dynamic_pointer_cast<Engine, IEngine>(in_engine);
  if (nullptr == engine) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_ENGINE,
               "The provided engine is not an onnxrt engine");
    return error;
  }

  auto model = std::dynamic_pointer_cast<onnxrt::Model, IModel>(in_model);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an onnxrt model");
    return error;
  }

  this->engine = engine;
  this->model = model;

  return error;
}

std::shared_ptr<r2i::IEngine> Parameters::GetEngine () {
  return this->engine;
}

Parameters::ParamDesc Parameters::Validate (const std::string &in_parameter,
    int type, const std::string &stype, RuntimeError &error) {
  ParamDesc undefined = {{.name = "", .description = ""}, nullptr};

  error.Clean ();

  auto match = this->parameter_map.find (in_parameter);

  /* The parameter wasn't found */
  if (match == this->parameter_map.end ()) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Parameter \"" + in_parameter + "\" does not exist");
    return undefined;
  }

  ParamDesc param = match->second;

  /* The parameter is not of the correct type */
  if (param.meta.type != type) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "Parameter \"" + in_parameter + "\" is not of type " + stype);
    return undefined;
  }

  return param;
}

RuntimeError Parameters::Get (const std::string &in_parameter, int &value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::INTEGER,
                                    "integer", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<IntAccessor>(param.accessor);

  error = accessor->Get ();
  if (error.IsError ()) {
    return error;
  }

  value = accessor->value;
  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter, double &value) {
  RuntimeError error;
  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::string &value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::STRING,
                                    "string", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<StringAccessor>(param.accessor);

  error = accessor->Get ();
  if (error.IsError ()) {
    return error;
  }

  value = accessor->value;
  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter, int in_value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::INTEGER,
                                    "integer", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<IntAccessor>(param.accessor);

  accessor->value = in_value;
  return accessor->Set ();
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              const std::string &in_value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::STRING,
                                    "string", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<StringAccessor>(param.accessor);

  accessor->value = in_value;
  return accessor->Set ();
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              double in_value) {
  RuntimeError error;
  return error;
}

RuntimeError Parameters::List (std::vector<ParameterMeta> &metas) {
  for (auto &param : this->parameter_map) {
    r2i::ParameterMeta meta = param.second.meta;
    metas.push_back(meta);
  }

  return RuntimeError();
}

}  // namespace onnxrt_openvino
}  // namespace r2i
