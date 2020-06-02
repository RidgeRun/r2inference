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

#ifndef R2I_EDGETPU_PARAMETERS_H
#define R2I_EDGETPU_PARAMETERS_H

#include <memory>
#include <string>
#include <unordered_map>

#include <r2i/iparameters.h>
#include <r2i/runtimeerror.h>
#include <r2i/tflite/engine.h>
#include <r2i/tflite/model.h>

namespace r2i {
namespace edgetpu {

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

};

}  // namespace edgetpu
}  // namespace r2i

#endif //R2I_EDGETPU_PARAMETERS_H
