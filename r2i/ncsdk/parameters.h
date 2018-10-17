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

#ifndef R2I_NCSDK_PARAMETERS_H
#define R2I_NCSDK_PARAMETERS_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <r2i/iparameters.h>
#include <r2i/ncsdk/engine.h>

namespace r2i {
namespace ncsdk {

class Parameters : public IParameters {
 public:
  Parameters ();

  RuntimeError Configure (std::shared_ptr<r2i::IEngine> in_engine,
                          std::shared_ptr<r2i::IModel> in_model) override;

  std::shared_ptr<r2i::IEngine> GetEngine () override;

  std::shared_ptr<r2i::IModel> GetModel ( ) override;

  RuntimeError Get (const std::string &in_parameter, int &value) override;

  RuntimeError Get (const std::string &in_parameter,
                    std::string &value) override;

  RuntimeError Set (const std::string &in_parameter,
                    const std::string &in_value) override;

  RuntimeError Set (const std::string &in_parameter, int in_value) override;

 private:
  std::shared_ptr<Engine> engine;
  std::shared_ptr<IModel> model;

  RuntimeError SetParameter (const std::unordered_map<std::string, int> &map,
                             const std::string &in_parameter,
                             const std::string &type,
                             const void *target,
                             unsigned int target_size);

  RuntimeError GetParameter (const std::unordered_map<std::string, int> &map,
                             const std::string &in_parameter,
                             const std::string &type,
                             void *target,
                             unsigned int *target_size);

  static RuntimeError SetParameterGlobal (Parameters *self, int param,
                                          void *target,
                                          unsigned int *target_size);

  static RuntimeError GetParameterGlobal (Parameters *self, int param,
                                          void *target,
                                          unsigned int *target_size);

  static RuntimeError SetParameterEngine (Parameters *self, int param,
                                          void *target,
                                          unsigned int *target_size);

  static RuntimeError GetParameterEngine (Parameters *self, int param,
                                          void *target,
                                          unsigned int *target_size);

  typedef std::function<RuntimeError(Parameters *, int param, void *target,
                                     unsigned int *target_size)> Accessor;



  typedef std::unordered_map<std::string, int> CodeMap;

  enum AccessorIndex {
    SET = 0,
    GET = 1,
  };

  struct AccessorNode {
    const CodeMap map;
    const Accessor accessor[2];
  };

  typedef std::vector<struct AccessorNode> AccessorVector;

  RuntimeError ApplyParameter (const AccessorVector &vec,
                               const std::string &in_parameter,
                               const std::string &type,
                               void *target,
                               unsigned int *target_size,
                               int accesor_index);

  const CodeMap parameter_map_global_string;
  const CodeMap parameter_map_global_int;
  const CodeMap parameter_map_device_int;
  const AccessorVector parameter_maps_int;
  const AccessorVector parameter_maps_string;
};

} // namespace ncsdk
} // namespace r2k

#endif //R2I_NCSDK_PARAMETERS_H
