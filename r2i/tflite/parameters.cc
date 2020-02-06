/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
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
namespace tflite {

#define PARAM(_name, _desc, _flags, _type, _acc) \
  {                           \
    (_name),                  \
    {                         \
      .meta = {               \
      .name = (_name),        \
      .description = (_desc), \
      .flags = (_flags),      \
      .type = (_type),        \
    },                        \
    .accessor = (_acc)        \
    }                         \
  }

Parameters::Parameters (): parameter_map ( {
  /* Model parameters */
  PARAM("number_of_threads", "Number of threads to run, greater than 0, or 0 to ignore",
        r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
        r2i::ParameterMeta::Type::INTEGER,
        std::make_shared<NumberOfThreadsAccessor>(this)),
  PARAM("allow_fp16", "Allow fp16 optimization, 1 to enable, 0 to disable",
        r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
        r2i::ParameterMeta::Type::INTEGER,
        std::make_shared<AllowFP16Accessor>(this)),
}
                                         ) {
}


RuntimeError Parameters::Configure (std::shared_ptr < r2i::IEngine >
                                    in_engine, std::shared_ptr < r2i::IModel > in_model) {
  RuntimeError error;

  if (nullptr == in_engine) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received null engine");
    return error;
  }

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received null model");
    return error;
  }

  auto engine = std::dynamic_pointer_cast < Engine, IEngine > (in_engine);
  if (nullptr == engine) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_ENGINE,
               "The provided engine is not an tflite engine");
    return error;
  }

  auto model = std::dynamic_pointer_cast < Model, IModel > (in_model);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tflite model");
    return error;
  }

  this->engine = engine;
  this->model = model;

  return error;
}

std::shared_ptr < r2i::IEngine > Parameters::GetEngine () {
  return this->engine;
}


std::shared_ptr < r2i::IModel > Parameters::GetModel () {
  return this->model;
}

Parameters::ParamDesc Parameters::
Validate (const std::string &in_parameter, int type,
          const std::string &stype, RuntimeError &error) {
  ParamDesc undefined = { {.name = "", .description = ""}, nullptr };

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

  auto accessor =
    std::dynamic_pointer_cast < IntAccessor > (param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tflite model");
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
                              std::string &value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::STRING, "string", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor =
    std::dynamic_pointer_cast < StringAccessor > (param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tflite model");
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

  auto accessor =
    std::dynamic_pointer_cast < StringAccessor > (param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tflite model");
    return error;
  }

  accessor->value = in_value;
  return accessor->Set ();
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              int in_value) {
  RuntimeError error;

  ParamDesc param = this->Validate (in_parameter,
                                    r2i::ParameterMeta::Type::INTEGER, "integer", error);
  if (error.IsError ()) {
    return error;
  }

  auto accessor =
    std::dynamic_pointer_cast < IntAccessor > (param.accessor);
  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an tflite model");
    return error;
  }

  accessor->value = in_value;
  return accessor->Set ();
}

RuntimeError Parameters::List (std::vector < ParameterMeta > &metas) {
  for (auto &param : this->parameter_map) {
    r2i::ParameterMeta meta = param.second.meta;
    metas.push_back (meta);
  }

  return RuntimeError ();
}

}// namespace tflite
}// namespace r2i
