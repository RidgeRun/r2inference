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
#include "r2i/ncsdk/parameteraccessors.h"

namespace r2i {
namespace ncsdk {

#define PARAM(_name, _desc, _flags, _type, _nccode, _set, _get) \
  {								\
    (_name),							\
    {								\
      .meta = {							\
	.name = (_name),					\
	.description = (_desc),					\
	.flags = (_flags),					\
	.type = (_type),					\
      },							\
      .nccode = (_nccode),					\
      .accessor = {						\
	(_set),							\
	(_get)							\
      }								\
    }								\
  }

Parameters::Parameters () :
  parameter_map ({
  /* Global parameters */
  PARAM("log-level", "NCSDK debug log level",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_LOG_LEVEL,
        &r2i::ncsdk::SetParameterGlobal,
        &r2i::ncsdk::GetParameterGlobal),

  /* Device parameters */
  PARAM("thermal-throttling-level", "Temp limit reched: 1) lower, 2) higher",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_THERMAL_THROTTLING_LEVEL,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("device-state", "The current state of the device: CREATED, OPENED, CLOSED, DESTROYED",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_STATE,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("current-memory-used", "Current device memory usage",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_CURRENT_MEMORY_USED,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("memory-size", "Device memory size",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_MEMORY_SIZE,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("max-fifo-num", "Max number of fifos supported",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_MAX_FIFO_NUM,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("allocated-fifo-num", "Current number of allocated fifos",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_ALLOCATED_FIFO_NUM,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("max-graph-num", "Max number of graphs supported",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_MAX_GRAPH_NUM,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("allocated-graph-num", "Current number of allocated graphs",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_ALLOCATED_GRAPH_NUM,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("option-class-limit", "Highest option class supported",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_OPTION_CLASS_LIMIT,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("max-executor-num", "Max numbers of executers per graph",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_DEVICE_MAX_EXECUTORS_NUM,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("device-debug-info", "Returns debug info",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::STRING, NC_RO_DEVICE_DEBUG_INFO,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),
  PARAM("device-name", "Returns the name of the device",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::STRING, NC_RO_DEVICE_NAME,
        &r2i::ncsdk::SetParameterEngine,
        &r2i::ncsdk::GetParameterEngine),

  /* Input fifo parameters */
  PARAM("input-fifo-type", "Fifo type from ncFifoType_t",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_FIFO_TYPE,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),
  PARAM("input-fifo-consumer-count",
        "Number of times an element must be read before removing from fifo. Defaults to 1.",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_FIFO_CONSUMER_COUNT,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),
  PARAM("input-fifo-data-type",
        "0) fp16 1) fp32. If selected fp32 the API will convert to fp16 automatically",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_FIFO_DATA_TYPE,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),
  PARAM("input-fifo-dont-block",
        "Don't block if the fifo is full (not implemented)",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_FIFO_DONT_BLOCK,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),
  PARAM("input-fifo-read-fill-level",
        "Current number of tensors in the read buffer",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_READ_FILL_LEVEL,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),
  PARAM("input-fifo-write-fill-level",
        "Current number of tensors in the write buffer",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_WRITE_FILL_LEVEL,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),
  PARAM("input-fifo-state",
        "The current fifo state: CREATED, ALLOCATED, DESTROYED",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_STATE,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),
  PARAM("input-fifo-element-data-size", "Element data size in bytes",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_ELEMENT_DATA_SIZE,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),
  PARAM("input-fifo-capacity", "The capacity of the input fifo",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_CAPACITY,
        &r2i::ncsdk::SetParameterInputFifo,
        &r2i::ncsdk::GetParameterInputFifo),

  /* Output fifo parameters */
  PARAM("output-fifo-type", "Fifo type from ncFifoType_t",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_FIFO_TYPE,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),
  PARAM("output-fifo-consumer-count",
        "Number of times an element must be read before removing from fifo. Defaults to 1.",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_FIFO_CONSUMER_COUNT,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),
  PARAM("output-fifo-data-type",
        "0) fp16 1) fp32. If selected fp32 the API will convert to fp16 automatically",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_FIFO_DATA_TYPE,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),
  PARAM("output-fifo-dont-block",
        "Don't block if the fifo is full (not implemented)",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_FIFO_DONT_BLOCK,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),
  PARAM("output-fifo-read-fill-level",
        "Current number of tensors in the read buffer",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_READ_FILL_LEVEL,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),
  PARAM("output-fifo-write-fill-level",
        "Current number of tensors in the write buffer",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_WRITE_FILL_LEVEL,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),
  PARAM("output-fifo-state",
        "The current fifo state: CREATED, ALLOCATED, DESTROYED",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_STATE,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),
  PARAM("output-fifo-element-data-size", "Element data size in bytes",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_ELEMENT_DATA_SIZE,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),
  PARAM("output-fifo-capacity", "The capacity of the output fifo",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_FIFO_CAPACITY,
        &r2i::ncsdk::SetParameterOutputFifo,
        &r2i::ncsdk::GetParameterOutputFifo),

  /* Graph parameters */
  PARAM("graph-state",
        "The current state of the graph: CREATED, ALLOCATED, WAITING_FOR_BUFFERS, RUNNING, DESTROYED",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_GRAPH_STATE,
        &r2i::ncsdk::SetParameterGraph,
        &r2i::ncsdk::GetParameterGraph),
  PARAM("graph-input-count", "Returns number of inputs, size of array returned",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_GRAPH_INPUT_COUNT,
        &r2i::ncsdk::SetParameterGraph,
        &r2i::ncsdk::GetParameterGraph),
  PARAM("graph-output-count",
        "Returns number of outputs, size of array returned",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_GRAPH_OUTPUT_COUNT,
        &r2i::ncsdk::SetParameterGraph,
        &r2i::ncsdk::GetParameterGraph),
  PARAM("graph-option-class-limit", "graph-option-class-limit",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::INTEGER, NC_RO_GRAPH_OPTION_CLASS_LIMIT,
        &r2i::ncsdk::SetParameterGraph,
        &r2i::ncsdk::GetParameterGraph),
  PARAM("graph-executors-num", "The amount of graph executors",
        r2i::ParameterMeta::Flags::READWRITE,
        r2i::ParameterMeta::Type::INTEGER, NC_RW_GRAPH_EXECUTORS_NUM,
        &r2i::ncsdk::SetParameterGraph,
        &r2i::ncsdk::GetParameterGraph),
  PARAM("graph-debug-info", "Returns debug info",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::STRING, NC_RO_GRAPH_DEBUG_INFO,
        &r2i::ncsdk::SetParameterGraph,
        &r2i::ncsdk::GetParameterGraph),
  PARAM("graph-name", "Returns the name of the graph",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::STRING, NC_RO_GRAPH_NAME,
        &r2i::ncsdk::SetParameterGraph,
        &r2i::ncsdk::GetParameterGraph),
  PARAM("graph-version", "Returns the version of the graph",
        r2i::ParameterMeta::Flags::READ,
        r2i::ParameterMeta::Type::STRING, NC_RO_GRAPH_VERSION,
        &r2i::ncsdk::SetParameterGraph,
        &r2i::ncsdk::GetParameterGraph),

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

  return this->ApplyParameter (this->parameter_map, in_parameter,
                               r2i::ParameterMeta::Type::INTEGER, "int",
                               &value,
                               &value_size, AccessorIndex::GET);
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::string &value) {
  RuntimeError error;
  unsigned int value_size = value.size();

  error = this->ApplyParameter (this->parameter_map, in_parameter,
                                r2i::ParameterMeta::Type::STRING, "string",
                                &(value[0]), &value_size, AccessorIndex::GET);

  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              const std::string &in_value) {
  unsigned int value_size = in_value.size() + 1;

  return this->ApplyParameter (this->parameter_map, in_parameter,
                               r2i::ParameterMeta::Type::STRING, "string",
                               const_cast<char *>(in_value.c_str()),
                               &value_size, AccessorIndex::SET);
}

RuntimeError Parameters::Set (const std::string &in_parameter, int in_value) {
  unsigned int value_size = sizeof (in_value);

  return this->ApplyParameter (this->parameter_map, in_parameter,
                               r2i::ParameterMeta::Type::INTEGER, "int",
                               &in_value,
                               &value_size, AccessorIndex::SET);
}

RuntimeError Parameters::ApplyParameter (const ParamMap &map,
    const std::string &in_parameter,
    const r2i::ParameterMeta::Type type,
    const std::string &stype,
    void *target,
    unsigned int *target_size,
    int accesor_index) {

  auto match = map.find (in_parameter);

  /* The parameter wasn't found */
  if (match == map.end ()) {

    return RuntimeError (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
                         "Parameter \""
                         + in_parameter + "\" does not exist");
  }

  ParamDesc param = match->second;

  /* The parameter is not of the correct type */
  if (param.meta.type != type) {
    return RuntimeError (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
                         "Parameter \""
                         + in_parameter + "\" is not of type " + stype);
  }

  /* Valid parameter found */
  Accessor apply = param.accessor[accesor_index];
  int nccode = param.nccode;

  return apply (this, nccode, target, target_size);
}

RuntimeError Parameters::ListParameters (std::vector<ParameterMeta> &metas) {
  for (auto &param : this->parameter_map) {
    r2i::ParameterMeta meta = param.second.meta;
    metas.push_back(meta);
  }

  return RuntimeError();
}

} // namespace ncsdk
} // namespace r2i
