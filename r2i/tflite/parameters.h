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

#ifndef R2I_TFLITE_PARAMETERS_H
#define R2I_TFLITE_PARAMETERS_H

#include <memory>
#include <string>
#include <unordered_map>

#include <r2i/iparameters.h>
#include <r2i/runtimeerror.h>
#include <r2i/tflite/engine.h>
#include <r2i/tflite/model.h>

namespace r2i {
namespace tflite {

class Parameters: public IParameters {
 public:
  Parameters ();

  RuntimeError Configure (std::shared_ptr < IEngine > in_engine,
                          std::shared_ptr < IModel > in_model) override;

  std::shared_ptr < IEngine > GetEngine () override;

  std::shared_ptr < IModel > GetModel () override;

  RuntimeError Get (const std::string &in_parameter, int &value) override;

  RuntimeError Get (const std::string &in_parameter, double &value) override;

  RuntimeError Get (const std::string &in_parameter,
                    std::string &value) override;

  RuntimeError Set (const std::string &in_parameter,
                    const std::string &in_value) override;

  RuntimeError Set (const std::string &in_parameter,
                    int in_value) override;

  RuntimeError Set (const std::string &in_parameter, double in_value) override;

  RuntimeError List (std::vector < ParameterMeta > &metas) override;

 private:
  std::shared_ptr < Engine > engine;
  std::shared_ptr < Model > model;

  friend class Accessor;

  class Accessor {
   public:
    Accessor (Parameters *target): target (target) {
    }
    virtual RuntimeError Set () = 0;
    virtual RuntimeError Get () = 0;
    virtual ~ Accessor () {
    }

   public:
    Parameters *target;
  };

  class StringAccessor: public Accessor {
   public:
    StringAccessor (Parameters *target): Accessor (target) {
    }
    std::string value;
  };

  class IntAccessor: public Accessor {
   public:
    IntAccessor (Parameters *target): Accessor (target) {}
    int value;
  };

  class DoubleAccessor : public Accessor {
   public:
    DoubleAccessor (Parameters *target) : Accessor(target) {}
    double value;
  };

  class VersionAccessor: public StringAccessor {
   public:
    VersionAccessor (Parameters *target): StringAccessor (target) {
    }
    RuntimeError Set () {
      return RuntimeError (RuntimeError::Code::WRONG_API_USAGE,
                           "Parameter is not writeable");
    }

    RuntimeError Get () {
      // TODO: Implement Version parameter
      return RuntimeError ();
    }
  };

  class NumberOfThreadsAccessor: public IntAccessor {
   public:
    NumberOfThreadsAccessor (Parameters *target): IntAccessor (target) {}
    RuntimeError Set () {
      return target->engine->SetNumberOfThreads(this->value);
    }
    RuntimeError Get () {
      this->value = target->engine->GetNumberOfThreads();
      return RuntimeError ();
    }
  };

  class AllowFP16Accessor: public IntAccessor {
   public:
    AllowFP16Accessor (Parameters *target): IntAccessor (target) {}
    RuntimeError Set () {
      return target->engine->SetAllowFP16(this->value);
    }
    RuntimeError Get () {
      this->value = target->engine->GetAllowFP16();
      return RuntimeError ();
    }
  };

  struct ParamDesc {
    ParameterMeta meta;
    std::shared_ptr < Accessor > accessor;
  };

  typedef std::unordered_map < std::string, ParamDesc > ParamMap;
  const ParamMap parameter_map;

  ParamDesc Validate (const std::string &in_parameter, int type,
                      const std::string &stype, RuntimeError &error);
};

}                             // namespace tflite
}                               // namespace r2i

#endif //R2I_TFLITE_PARAMETERS_H
