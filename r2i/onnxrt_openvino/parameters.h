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

#ifndef R2I_ONNXRT_OPENVINO_PARAMETERS_H
#define R2I_ONNXRT_OPENVINO_PARAMETERS_H

#include <memory>
#include <string>
#include <unordered_map>

#include <r2i/iparameters.h>
#include <r2i/runtimeerror.h>
#include <r2i/onnxrt_openvino/engine.h>
#include <r2i/onnxrt/parameters.h>

namespace r2i {
namespace onnxrt_openvino {

class Parameters: public r2i::onnxrt::Parameters {
 public:
  Parameters ();
  RuntimeError Configure (std::shared_ptr<IEngine> in_engine,
                          std::shared_ptr<IModel> in_model) override;
  std::shared_ptr<IEngine> GetEngine () override;
  RuntimeError Get (const std::string &in_parameter, int &value) override;
  RuntimeError Get (const std::string &in_parameter, double &value) override;
  RuntimeError Get (const std::string &in_parameter,
                    std::string &value) override;
  RuntimeError Set (const std::string &in_parameter,
                    const std::string &in_value) override;
  RuntimeError Set (const std::string &in_parameter, int in_value) override;
  RuntimeError Set (const std::string &in_parameter, double in_value) override;
  RuntimeError List (std::vector<ParameterMeta> &metas) override;

 private:
  std::shared_ptr <Engine> engine;

  friend class Accessor;

  class Accessor {
   public:
    Accessor (Parameters *target) : target(target) {}
    virtual RuntimeError Set () = 0;
    virtual RuntimeError Get () = 0;
    virtual ~Accessor() {}

   public:
    Parameters *target;
  };

  class StringAccessor : public Accessor {
   public:
    StringAccessor (Parameters *target) : Accessor(target) {}
    std::string value;
  };

  class IntAccessor : public Accessor {
   public:
    IntAccessor (Parameters *target) : Accessor(target) {}
    int value;
  };

  class LoggingLevelAccessor : public IntAccessor {
   public:
    LoggingLevelAccessor (Parameters *target) : IntAccessor(target) {}

    RuntimeError Set () {
      return target->engine->SetLoggingLevel(this->value);
    }

    RuntimeError Get () {
      this->value = target->engine->GetLoggingLevel();
      return RuntimeError ();
    }
  };

  class LogIdAccessor : public StringAccessor {
   public:
    LogIdAccessor (Parameters *target) : StringAccessor(target) {}
    RuntimeError Set () {
      return target->engine->SetLogId(this->value);
    }

    RuntimeError Get () {
      this->value = target->engine->GetLogId();
      return RuntimeError ();
    }
  };

  class IntraNumThreadsAccessor : public IntAccessor {
   public:
    IntraNumThreadsAccessor (Parameters *target) : IntAccessor(target) {}

    RuntimeError Set () {
      return target->engine->SetIntraNumThreads(this->value);
    }

    RuntimeError Get () {
      this->value = target->engine->GetIntraNumThreads();
      return RuntimeError ();
    }
  };

  class GraphOptLevelAccessor : public IntAccessor {
   public:
    GraphOptLevelAccessor (Parameters *target) : IntAccessor(target) {}

    RuntimeError Set () {
      return target->engine->SetGraphOptLevel(this->value);
    }

    RuntimeError Get () {
      this->value = target->engine->GetGraphOptLevel();
      return RuntimeError ();
    }
  };

  class HardwareIdAccessor : public StringAccessor {
   public:
    HardwareIdAccessor (Parameters *target) : StringAccessor(target) {}
    RuntimeError Set () {
      return target->engine->SetHardwareId(this->value);
    }

    RuntimeError Get () {
      this->value = target->engine->GetHardwareId();
      return RuntimeError ();
    }
  };

  struct ParamDesc {
    ParameterMeta meta;
    std::shared_ptr<Accessor> accessor;
  };

  typedef std::unordered_map<std::string, ParamDesc> ParamMap;
  ParamMap parameter_map;

  ParamDesc Validate (const std::string &in_parameter, int type,
                      const std::string &stype, RuntimeError &error);

  ParameterMeta hardware_id_meta = {
    .name = "hardware-id",
    .description = "OpenVINO hardware device id",
    .flags = r2i::ParameterMeta::Flags::READWRITE | r2i::ParameterMeta::Flags::WRITE_BEFORE_START,
    .type = r2i::ParameterMeta::Type::STRING
  };
};

}  // namespace onnxrt_openvino
}  // namespace r2i

#endif //R2I_ONNXRT_OPENVINO_PARAMETERS_H
