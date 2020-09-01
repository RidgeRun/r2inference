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

#ifndef R2I_ONNXRT_PARAMETERS_H
#define R2I_ONNXRT_PARAMETERS_H

#include <memory>
#include <string>
#include <unordered_map>

#include <r2i/iparameters.h>
#include <r2i/runtimeerror.h>
#include <r2i/onnxrt/engine.h>
#include <r2i/onnxrt/model.h>
#include <r2i/onnxrt/parameteraccessors.h>

namespace r2i {
namespace onnxrt {

class Parameters: public IParameters {
 public:
  Parameters ();
  RuntimeError Configure (std::shared_ptr<IEngine> in_engine,
                          std::shared_ptr<IModel> in_model) override;
  std::shared_ptr<IEngine> GetEngine () override;
  std::shared_ptr<IModel> GetModel ( ) override;
  RuntimeError Get (const std::string &in_parameter, int &value) override;
  RuntimeError Get (const std::string &in_parameter, double &value) override;
  RuntimeError Get (const std::string &in_parameter,
                    std::string &value) override;
  RuntimeError Set (const std::string &in_parameter,
                    const std::string &in_value) override;
  RuntimeError Set (const std::string &in_parameter, int in_value) override;
  RuntimeError Set (const std::string &in_parameter, double in_value) override;
  RuntimeError List (std::vector<ParameterMeta> &metas) override;
  /* To set/get Parameters of the internal Engine instance */
  RuntimeError SetLogLevel (int value);
  RuntimeError GetLogLevel (int &value);
  RuntimeError SetIntraNumThreads (int value);
  RuntimeError GetIntraNumThreads (int &value);
  RuntimeError SetGraphOptLevel (int value);
  RuntimeError GetGraphOptLevel (int &value);
  RuntimeError SetLogId (const std::string &value);
  RuntimeError GetLogId (std::string &value);

 protected:
  std::shared_ptr <Engine> engine;
  std::shared_ptr <IModel> model;
  std::shared_ptr<r2i::onnxrt::Accessor> accessor;

  struct ParamDesc {
    ParameterMeta meta;
    std::shared_ptr<Accessor> accessor;
  };

  typedef std::unordered_map<std::string, ParamDesc> ParamMap;
  ParamMap parameter_map;

  ParamDesc Validate (const std::string &in_parameter, int type,
                      const std::string &stype, RuntimeError &error);

  ParameterMeta logging_level_meta = {
    .name = "logging-level",
    .description = "ONNXRT Logging Level",
    .flags = r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
    .type = r2i::ParameterMeta::Type::INTEGER
  };
  ParameterMeta log_id_meta = {
    .name = "log-id",
    .description = "String identification for ONNXRT environment",
    .flags = r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
    .type = r2i::ParameterMeta::Type::STRING
  };
  ParameterMeta intra_num_threads_meta = {
    .name = "intra-num-threads",
    .description = "Number of threads to parallelize execution within model nodes",
    .flags = r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
    .type = r2i::ParameterMeta::Type::INTEGER
  };
  ParameterMeta graph_optimization_level_meta = {
    .name = "graph-optimization-level",
    .description = "ONNXRT graph optimization level",
    .flags = r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
    .type = r2i::ParameterMeta::Type::INTEGER
  };

};

}  // namespace onnxrt
}  // namespace r2i

#endif //R2I_ONNXRT_PARAMETERS_H
