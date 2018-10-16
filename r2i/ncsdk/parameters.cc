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

#include <cstring>
#include <memory>
#include <mvnc.h>

#include "r2i/ncsdk/parameters.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

static const std::unordered_map<std::string, int> parameter_int_map ({
  {"log-level", NC_RW_LOG_LEVEL},
  {"api-version", NC_RO_API_VERSION}, //for testing purposes
});

static const std::unordered_map<std::string, int> parameter_string_map ({
});

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
               "The provided engine is not an NCSDK engine");
    return error;
  }

  this->engine = engine;
  this->model = in_model;

  return error;
}

std::shared_ptr<r2i::IEngine> Parameters::GetEngine () {
  return this->engine;
}


std::shared_ptr<r2i::IModel> Parameters::GetModel () {
  return this->model;
}

RuntimeError Parameters::Get (const std::string &in_parameter, int &value) {
  unsigned int value_size = sizeof (value);

  return this->GetParameter (parameter_int_map, in_parameter, "int", &value,
                             &value_size);
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::string &value) {
  unsigned int value_size = value.size();

  return this->GetParameter (parameter_string_map, in_parameter, "int",
                             &(value[0]),
                             &value_size);
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              const std::string &in_value) {
  return this->SetParameter (parameter_string_map, in_parameter, "string",
                             in_value.c_str(), in_value.size() + 1);
}

RuntimeError Parameters::Set (const std::string &in_parameter, int in_value) {
  return this->SetParameter (parameter_int_map, in_parameter, "int", &in_value,
                             sizeof (in_value));
}

RuntimeError Parameters::InteractWithParameter (const
    std::unordered_map<std::string, int> &map, const std::string &in_parameter,
    const std::string &type, void *target, unsigned int *target_size,
    param_apply apply) {
  RuntimeError error;

  auto search = map.find (in_parameter);

  if (map.end () == search) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, "Parameter \""
               + in_parameter + "\" does not exist or is not of " + type + " type");
    return error;
  }

  ncStatus_t ret = apply (search->second, target, target_size);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               GetStringFromStatus (ret, error));
  }

  return error;
}

RuntimeError Parameters::SetParameter (const
                                       std::unordered_map<std::string, int> &map,
                                       const std::string &in_parameter,
                                       const std::string &type,
                                       const void *target, unsigned int
                                       target_size) {

  param_apply apply = [] (int param, void *target,
  unsigned int *target_size) -> ncStatus_t {
    return ncGlobalSetOption (param, target, *target_size);
  };

  /* Allow this const cast-away to favor code reusability, we are the
     only ones who interact with target in the provided lambda
  */
  return this->InteractWithParameter (map, in_parameter, type,
                                      const_cast<void *> (target), &target_size, apply);
}

RuntimeError Parameters::GetParameter (const
                                       std::unordered_map<std::string, int> &map,
                                       const std::string &in_parameter,
                                       const std::string &type, void *target,
                                       unsigned int *target_size) {
  param_apply apply = [] (int param, void *target,
  unsigned int *target_size) -> ncStatus_t {
    return ncGlobalGetOption (param, target, target_size);
  };

  return this->InteractWithParameter (map, in_parameter, type, target,
                                      target_size, apply);
}

} // namespace ncsdk
} // namespace r2i
