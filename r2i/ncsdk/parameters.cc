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
parameter_map_input_fifo_int ({
  {"input-fifo-type", NC_RW_FIFO_TYPE},
  {"input-fifo-consumer-count", NC_RW_FIFO_CONSUMER_COUNT},
  {"input-fifo-data-type", NC_RW_FIFO_DATA_TYPE},
  {"input-fifo-dont-block", NC_RW_FIFO_DONT_BLOCK},
  {"input-fifo-capacity", NC_RO_FIFO_CAPACITY},
  {"input-fifo-read-fill-level", NC_RO_FIFO_READ_FILL_LEVEL},
  {"input-fifo-write-fill-level", NC_RO_FIFO_WRITE_FILL_LEVEL},
  {"input-fifo-graph-tensor-descriptor", NC_RO_FIFO_GRAPH_TENSOR_DESCRIPTOR},
  {"input-fifo-state", NC_RO_FIFO_STATE},
  {"input-fifo-element-data-size", NC_RO_FIFO_ELEMENT_DATA_SIZE}
}),
parameter_map_output_fifo_int ({
  {"output-fifo-type", NC_RW_FIFO_TYPE},
  {"output-fifo-consumer-count", NC_RW_FIFO_CONSUMER_COUNT},
  {"output-fifo-data-type", NC_RW_FIFO_DATA_TYPE},
  {"output-fifo-dont-block", NC_RW_FIFO_DONT_BLOCK},
  {"output-fifo-capacity", NC_RO_FIFO_CAPACITY},
  {"output-fifo-read-fill-level", NC_RO_FIFO_READ_FILL_LEVEL},
  {"output-fifo-write-fill-level", NC_RO_FIFO_WRITE_FILL_LEVEL},
  {"output-fifo-graph-tensor-descriptor", NC_RO_FIFO_GRAPH_TENSOR_DESCRIPTOR},
  {"output-fifo-state", NC_RO_FIFO_STATE},
  {"output-fifo-element-data-size", NC_RO_FIFO_ELEMENT_DATA_SIZE}
}),
parameter_maps_int {
  {this->parameter_map_global_int, {&r2i::ncsdk::Parameters::SetParameterGlobal, &r2i::ncsdk::Parameters::GetParameterGlobal}},
  {this->parameter_map_device_int, {&r2i::ncsdk::Parameters::SetParameterEngine, &r2i::ncsdk::Parameters::GetParameterEngine}},
  {this->parameter_map_input_fifo_int, {&r2i::ncsdk::Parameters::SetParameterInputFifo, &r2i::ncsdk::Parameters::GetParameterInputFifo}},
  {this->parameter_map_output_fifo_int, {&r2i::ncsdk::Parameters::SetParameterOutputFifo, &r2i::ncsdk::Parameters::GetParameterOutputFifo}}} {
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

  for (auto &accessmap : vec) {
    auto param = accessmap.map.find (in_parameter);

    /* Parameter found in current map, process it */
    if (param != accessmap.map.end ()) {
      Accessor apply = accessmap.accessor[accesor_index];
      int ncparam = param->second;

      return apply (this, ncparam, target, target_size);
    }
  }

  /* The parameter wasn't found in any map */
  return RuntimeError (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
                       "Parameter \""
                       + in_parameter + "\" does not exist or is not of " + type + " type");
}

static RuntimeError ValidateAccessorParameters (Parameters *self, void *target,
    unsigned int *target_size) {
  RuntimeError error;

  if (nullptr == self) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "NULL instance, something really bad is happening");
    return error;
  }

  if (nullptr == target) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "NULL target passed to accessor");
    return error;
  }

  if (nullptr == target_size) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "NULL size passed to accessor");
    return error;
  }

  return error;
}

RuntimeError Parameters::SetParameterGlobal (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  ncStatus_t ncret = ncGlobalSetOption (param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError Parameters::GetParameterGlobal (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  ncStatus_t ncret = ncGlobalGetOption (param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

static RuntimeError ValidateEngineAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  std::shared_ptr<IEngine> engine = self->GetEngine();
  if (nullptr == engine) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "Parameters not been configured with a valid engine");
    return error;
  }

  return error;
}

static RuntimeError ValidateEngineDeviceAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  std::shared_ptr<IEngine> engine = self->GetEngine();
  ncDeviceHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                             (engine)->GetDeviceHandler ();
  if (nullptr == handle) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "No NCSDK device configured");
    return error;
  }

  return error;
}

RuntimeError Parameters::SetParameterEngine (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineDeviceAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncDeviceHandle_t *handle = self->engine->GetDeviceHandler ();
  ncStatus_t ncret = ncDeviceSetOption (handle, param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError Parameters::GetParameterEngine (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineDeviceAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncDeviceHandle_t *handle = self->engine->GetDeviceHandler ();
  ncStatus_t ncret = ncDeviceGetOption (handle, param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

static RuntimeError ValidateInputFifoAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  std::shared_ptr<IEngine> engine = self->GetEngine();

  ncFifoHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                           (engine)->GetInputFifoHandler ();
  if (nullptr == handle) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "No NCSDK input fifo configured");
    return error;
  }

  return error;
}

RuntimeError Parameters::SetParameterInputFifo (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateInputFifoAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = self->engine->GetInputFifoHandler ();
  ncStatus_t ncret = ncFifoSetOption (handle, param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError Parameters::GetParameterInputFifo (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateInputFifoAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = self->engine->GetInputFifoHandler ();
  ncStatus_t ncret = ncFifoGetOption (handle, param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

static RuntimeError ValidateOutputFifoAccessorParameters (Parameters *self,
    void *target, unsigned int *target_size) {
  RuntimeError error;

  error = ValidateEngineAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  std::shared_ptr<IEngine> engine = self->GetEngine();

  ncFifoHandle_t *handle = std::dynamic_pointer_cast<Engine, IEngine>
                           (engine)->GetOutputFifoHandler ();
  if (nullptr == handle) {
    error.Set (r2i::RuntimeError::Code::NULL_PARAMETER,
               "No NCSDK output fifo configured");
    return error;
  }

  return error;
}

RuntimeError Parameters::SetParameterOutputFifo (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateOutputFifoAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = self->engine->GetOutputFifoHandler ();
  ncStatus_t ncret = ncFifoSetOption (handle, param, target, *target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

RuntimeError Parameters::GetParameterOutputFifo (Parameters *self, int param,
    void *target,
    unsigned int *target_size) {
  RuntimeError error;

  error = ValidateOutputFifoAccessorParameters (self, target, target_size);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    return error;
  }

  /* Valid handle has already been validated with the method above */
  ncFifoHandle_t *handle = self->engine->GetOutputFifoHandler ();
  ncStatus_t ncret = ncFifoGetOption (handle, param, target, target_size);
  if (NC_OK != ncret) {
    error.Set (r2i::RuntimeError::Code::FRAMEWORK_ERROR, GetStringFromStatus (ncret,
               error));
    return error;
  }

  return error;
}

} // namespace ncsdk
} // namespace r2i
