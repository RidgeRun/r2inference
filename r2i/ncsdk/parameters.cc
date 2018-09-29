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

#include "r2i/ncsdk/parameters.h"

namespace r2i {
namespace ncsdk {

void Parameters::Configure (std::shared_ptr<r2i::IEngine> in_engine,
                            std::shared_ptr<r2i::IModel> in_model,
                            RuntimeError &error) {
  error.Clean ();

  if (nullptr == in_engine) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received null engine");
    return;
  }

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received null model");
    return;
  }
}

void Parameters::Get (const std::string in_parameter, int &value,
                      r2i::RuntimeError &error ) {}

void Parameters::Get (const std::string in_parameter, const std::string &value,
                      r2i::RuntimeError &error ) {}

void Parameters::Set (const std::string in_parameter,
                      const std::string &in_value,
                      RuntimeError &error ) {}

void Parameters::Set (const std::string in_parameter, int in_value,
                      RuntimeError &error ) {}

}
}
