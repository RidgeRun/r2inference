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
namespace onnxrt {

Parameters::Parameters () {
  ParamDesc logging_level_desc = {
    {logging_level_meta},
    std::make_shared<r2i::onnxrt::LoggingLevelAccessor>()
  };
  parameter_map.emplace(std::make_pair(logging_level_meta.name,
                                       logging_level_desc));

  ParamDesc log_id_desc = {
    {log_id_meta},
    std::make_shared<r2i::onnxrt::LogIdAccessor>()
  };
  parameter_map.emplace(std::make_pair(log_id_meta.name, log_id_desc));

  ParamDesc intra_num_threads_desc = {
    {intra_num_threads_meta},
    std::make_shared<r2i::onnxrt::IntraNumThreadsAccessor>()
  };
  parameter_map.emplace(std::make_pair(intra_num_threads_meta.name,
                                       intra_num_threads_desc));

  ParamDesc graph_optimization_level_desc = {
    {graph_optimization_level_meta},
    std::make_shared<r2i::onnxrt::GraphOptLevelAccessor>()
  };
  parameter_map.emplace(std::make_pair(graph_optimization_level_meta.name,
                                       graph_optimization_level_desc));
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
               "The provided engine is not an ONNXRT engine");
    return error;
  }

  auto model = std::dynamic_pointer_cast<Model, IModel>(in_model);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an ONNXRT model");
    return error;
  }

  this->engine = engine;
  this->model = model;

  return error;
}

std::shared_ptr<r2i::IEngine> Parameters::GetEngine () {
  return this->engine;
}

std::shared_ptr<r2i::IModel> Parameters::GetModel () {
  return this->model;
}

RuntimeError Parameters::Get (const std::string &in_parameter, int &value) {
  RuntimeError error;

  /* Return parameter that matches the name */
  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::INTEGER,
                                    "integer", error);
  if (error.IsError ()) {
    return error;
  }

  /* Valid parameter found */
  auto accessor = std::dynamic_pointer_cast<IntAccessor>(param.accessor);

  error = accessor->Get (*this);
  if (error.IsError ()) {
    return error;
  }

  value = accessor->value;

  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter, double &value) {
  RuntimeError error;
  error.Set(RuntimeError::NOT_IMPLEMENTED,
            "Parameters::Get (double) method not implemented");
  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::string &value) {
  RuntimeError error;

  /* Return parameter that matches the name */
  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::STRING,
                                    "string", error);
  if (error.IsError ()) {
    return error;
  }

  /* Valid parameter found */
  auto accessor = std::dynamic_pointer_cast<StringAccessor>(param.accessor);

  error = accessor->Get (*this);
  if (error.IsError ()) {
    return error;
  }

  value = accessor->value;
  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter, int in_value) {
  RuntimeError error;

  /* Return parameter that matches the name */
  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::INTEGER,
                                    "integer", error);
  if (error.IsError ()) {
    return error;
  }

  /* Valid parameter found */
  auto accessor = std::dynamic_pointer_cast<IntAccessor>(param.accessor);
  accessor->value = in_value;
  error = accessor->Set (*this);

  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              const std::string &in_value) {
  RuntimeError error;

  /* Return parameter that matches the name */
  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::STRING,
                                    "string", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<StringAccessor>(param.accessor);

  accessor->value = in_value;
  return accessor->Set (*this);
  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              double in_value) {
  RuntimeError error;
  error.Set(RuntimeError::NOT_IMPLEMENTED,
            "Parameters::Set (double) method not implemented");
  return error;
}

RuntimeError Parameters::List (std::vector<ParameterMeta> &metas) {
  for (auto &param : this->parameter_map) {
    r2i::ParameterMeta meta = param.second.meta;
    metas.push_back(meta);
  }

  return RuntimeError();
}

Parameters::ParamDesc Parameters::Validate (const std::string &in_parameter,
    int type, const std::string &stype,
    RuntimeError &error) {
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

RuntimeError Parameters::SetLogLevel (int value) {
  RuntimeError error;
  error = this->engine->SetLoggingLevel(value);
  return error;
}

RuntimeError Parameters::GetLogLevel (int &value) {
  RuntimeError error;
  value = this->engine->GetLoggingLevel();
  return error;
}

RuntimeError Parameters::SetIntraNumThreads (int value) {
  RuntimeError error;
  error = this->engine->SetIntraNumThreads(value);
  return error;
}

RuntimeError Parameters::GetIntraNumThreads (int &value) {
  RuntimeError error;
  value = this->engine->GetIntraNumThreads();
  return error;
}

RuntimeError Parameters::SetGraphOptLevel (int value) {
  RuntimeError error;
  error = this->engine->SetGraphOptLevel(value);
  return error;
}

RuntimeError Parameters::GetGraphOptLevel (int &value) {
  RuntimeError error;
  value = this->engine->GetGraphOptLevel();
  return error;
}

RuntimeError Parameters::SetLogId (const std::string &value) {
  RuntimeError error;
  error = this->engine->SetLogId(value);
  return error;
}

RuntimeError Parameters::GetLogId (std::string &value) {
  RuntimeError error;
  value = this->engine->GetLogId();
  return error;
}

}  // namespace onnxrt
}  // namespace r2i
