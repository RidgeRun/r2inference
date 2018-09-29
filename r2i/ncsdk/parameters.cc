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

#include <mvnc.h>
#include <unordered_map>

#include "r2i/ncsdk/parameters.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

static const std::unordered_map<std::string, int> parameter_int_map ({
  {"log-level", NC_RW_LOG_LEVEL},
  {"api-version", NC_RO_API_VERSION},
});

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

  this->engine = in_engine;
  this->model = in_model;
}

std::shared_ptr<r2i::IEngine> Parameters::GetEngine (RuntimeError &error) {
  error.Clean ();

  return this->engine;
}


std::shared_ptr<r2i::IModel> Parameters::GetModel (RuntimeError &error) {
  error.Clean ();

  return this->model;
}

void Parameters::Get (const std::string in_parameter, int &value,
                      r2i::RuntimeError &error ) {}

void Parameters::Get (const std::string in_parameter, const std::string &value,
                      r2i::RuntimeError &error ) {}

void Parameters::Set (const std::string in_parameter,
                      const std::string &in_value,
                      RuntimeError &error ) {}

void Parameters::Set (const std::string &in_parameter, int in_value,
                      RuntimeError &error ) {
  error.Clean ();

  auto search = parameter_int_map.find (in_parameter);

  if (parameter_int_map.end () == search) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER, "Parameter \""
               + in_parameter + "\" does not exist or is not of integer type");
    return;
  }

  ncStatus_t ret = ncGlobalSetOption(search->second, &in_value,
                                     sizeof (in_value));
  if (NC_OK != ret) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               GetStringFromStatus (ret, error));
  }
}

}
}
