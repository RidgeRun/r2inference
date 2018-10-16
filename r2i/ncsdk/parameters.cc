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

Parameters::Parameters () :
  parameter_map_global_string ({
}),
parameter_map_global_int ({
  {"log-level", NC_RW_LOG_LEVEL},
  {"api-version", NC_RO_API_VERSION},
}),
parameter_map_device_int ({
  {"thermal-throttling-level", NC_RO_DEVICE_THERMAL_THROTTLING_LEVEL},
  {"device-state", NC_RO_DEVICE_STATE},
  {"current-memory-used", NC_RO_DEVICE_CURRENT_MEMORY_USED},
  {"memory-size", NC_RO_DEVICE_MEMORY_SIZE},
  {"max-fifo-num", NC_RO_DEVICE_MAX_FIFO_NUM},
  {"allocated-fifo-num", NC_RO_DEVICE_ALLOCATED_FIFO_NUM},
  {"max-graph-num", NC_RO_DEVICE_MAX_GRAPH_NUM},
  {"allocated-graph-num", NC_RO_DEVICE_ALLOCATED_GRAPH_NUM},
  {"option-class-limit", NC_RO_DEVICE_OPTION_CLASS_LIMIT},
  {"max-executor-num", NC_RO_DEVICE_MAX_EXECUTORS_NUM},
}),
parameter_maps_int {
  {this->parameter_map_global_int, {&r2i::ncsdk::Parameters::SetParameterGlobal, &r2i::ncsdk::Parameters::GetParameterGlobal}}
} {
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

  return this->ApplyParameter (this->parameter_maps_int, in_parameter, "int",
                               &value,
                               &value_size, AccessorIndex::GET);
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::string &value) {
  unsigned int value_size = value.size();

  return this->ApplyParameter (this->parameter_maps_int, in_parameter, "int",
                               &(value[0]),
                               &value_size, AccessorIndex::GET);
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              const std::string &in_value) {
  unsigned int value_size = in_value.size() + 1;

  return this->ApplyParameter (this->parameter_maps_string, in_parameter,
                               "string", const_cast<char *>(in_value.c_str()),
                               &value_size, AccessorIndex::SET);
}

RuntimeError Parameters::Set (const std::string &in_parameter, int in_value) {
  unsigned int value_size = sizeof (in_value);

  return this->ApplyParameter (this->parameter_maps_int, in_parameter, "int",
                               &in_value,
                               &value_size, AccessorIndex::SET);
}

RuntimeError Parameters::ApplyParameter (const AccessorVector &vec,
    const std::string &in_parameter,
    const std::string &type,
    void *target,
    unsigned int *target_size,
    int accesor_index) {
  RuntimeError error;

  for (auto &accessmap : vec) {
    auto param = accessmap.map.find (in_parameter);

    /* Parameter found in current map, process it */
    if (param != accessmap.map.end ()) {
      Accessor apply = accessmap.accessor[accesor_index];
      int ncparam = param->second;

      ncStatus_t ret = apply (this, ncparam, target, target_size);
      if (NC_OK != ret) {
        error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
                   GetStringFromStatus (ret, error));
      }

      return error;
    }
  }

  /* The parameter wasn't found in any map */
  error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, "Parameter \""
             + in_parameter + "\" does not exist or is not of " + type + " type");
  return error;
}

ncStatus_t Parameters::SetParameterGlobal (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  return ncGlobalSetOption (param, target, *target_size);
}

ncStatus_t Parameters::GetParameterGlobal (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  return ncGlobalGetOption (param, target, target_size);
}

} // namespace ncsdk
} // namespace r2i
