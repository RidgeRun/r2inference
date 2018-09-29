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

#include <r2i/iparameters.h>

namespace r2i {
namespace ncsdk {

class Parameters : public IParameters {
 public:
  virtual void Configure (std::shared_ptr<r2i::IEngine> in_engine,
                          std::shared_ptr<r2i::IModel> in_model,
                          RuntimeError &error) override;

  virtual void Get (const std::string in_parameter, int &value,
                    r2i::RuntimeError &error ) override;

  virtual void Get (const std::string in_parameter, const std::string &value,
                    r2i::RuntimeError &error ) override;

  virtual void Set (const std::string in_parameter, const std::string &in_value,
                    RuntimeError &error ) override;

  virtual void Set (const std::string in_parameter, int in_value,
                    RuntimeError &error ) override;
};

} // namespace ncsdk
} // namespace r2k

#endif //R2I_NCSDK_PARAMETERS_H
