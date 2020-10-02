/* Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
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
#include <tensorflow/c/c_api.h>

namespace r2i {
namespace tensorflow {

#define PARAM(_name, _desc, _flags, _type, _acc) \
  {						 \
    (_name),					 \
    {						 \
      .meta = {					 \
	.name = (_name),			 \
	.description = (_desc),			 \
	.flags = (_flags),			 \
	.type = (_type),			 \
      },					 \
      .accessor = (_acc)			 \
    }						 \
  }

Parameters::Parameters () :
  parameter_map ({
  /* Global parameters */
  PARAM("version", "Tensorflow version",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::STRING,
        std::make_shared<VersionAccessor>(this)),
  PARAM("gpu-memory-usage", "Per process GPU memory usage fraction",
        r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
        r2i::ParameterMeta::Type::DOUBLE,
        std::make_shared<MemoryUsageAccessor>(this)),

  /* Model parameters */
  PARAM("input-layer", "Name of the input layer in the graph",
        r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
        r2i::ParameterMeta::Type::STRING,
        std::make_shared<InputLayerAccessor>(this)),
  PARAM("output-layer", "Name of the output layer in the graph",
        r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
        r2i::ParameterMeta::Type::STRING,
        std::make_shared<OutputLayerAccessor>(this)),
  PARAM("output-layers", "Names of the output layers in the graph",
        r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
        r2i::ParameterMeta::Type::VECTOR,
        std::make_shared<OutputLayersAccessor>(this)),
}) {
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
               "The provided engine is not an tensorflow engine");
    return error;
  }

  auto model = std::dynamic_pointer_cast<Model, IModel>(in_model);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tensorflow model");
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
                                    r2i::ParameterMeta::Type::INTEGER, "integer", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<IntAccessor>(param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tensorflow model");
    return error;
  }

  error = accessor->Get ();
  if (error.IsError ()) {
    return error;
  }

  value = accessor->value;
  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter, double &value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::DOUBLE, "double", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<DoubleAccessor>(param.accessor);

  error = accessor->Get ();
  if (error.IsError ()) {
    return error;
  }

  value = accessor->value;
  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::string &value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::STRING, "string", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<StringAccessor>(param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tensorflow model");
    return error;
  }

  error = accessor->Get ();
  if (error.IsError ()) {
    return error;
  }

  value = accessor->value;
  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::vector< std::string > &value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::VECTOR, "vector", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<VectorAccessor>(param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tensorflow model");
    return error;
  }

  error = accessor->Get ();
  if (error.IsError ()) {
    return error;
  }

  value = accessor->value;
  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              const std::string &in_value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::STRING, std::string ("string"), error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<StringAccessor>(param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tensorflow model");
    return error;
  }

  accessor->value = in_value;
  return accessor->Set ();
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              double in_value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::DOUBLE, "double", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<DoubleAccessor>(param.accessor);

  accessor->value = in_value;
  return accessor->Set ();
}

RuntimeError Parameters::Set (const std::string &in_parameter, int in_value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::INTEGER, "integer", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<IntAccessor>(param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tensorflow model");
    return error;
  }

  accessor->value = in_value;
  return accessor->Set ();
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              std::vector< std::string > value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::VECTOR, "vector", error);

  if (error.IsError ()) {
    return error;
  }

  auto accessor = std::dynamic_pointer_cast<VectorAccessor>(param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tensorflow model");
    return error;
  }

  accessor->value = value;
  return accessor->Set ();
}

RuntimeError Parameters::List (std::vector<ParameterMeta> &metas) {
  for (auto &param : this->parameter_map) {
    r2i::ParameterMeta meta = param.second.meta;
    metas.push_back(meta);
  }

  return RuntimeError();
}

} // namespace tensorflow
} // namespace r2i
