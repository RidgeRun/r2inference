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

#include <functional>
#include <mvnc.h>

#include "r2i/ncsdk/parameters.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

static const std::unordered_map<std::string, int> parameter_int_map ({
  {"log-level", NC_RW_LOG_LEVEL},
  {"api-version", NC_RO_API_VERSION},
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

  this->engine = in_engine;
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
  RuntimeError error;

  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::string &value) {
  RuntimeError error;

  return error;
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

RuntimeError Parameters::SetParameter (const
                                       std::unordered_map<std::string, int> &map,
                                       const std::string &in_parameter,
                                       const std::string &type,
                                       const void *target,
                                       unsigned int target_size) {
  RuntimeError error;

  auto search = map.find (in_parameter);

  if (map.end () == search) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, "Parameter \""
               + in_parameter + "\" does not exist or is not of " + type + " type");
    return error;
  }

  ncStatus_t ret = ncGlobalSetOption (search->second, target, target_size);
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               GetStringFromStatus (ret, error));
  }

  return error;
}

}
}
