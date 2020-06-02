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

#include "parameters.h"

namespace r2i {
namespace edgetpu {

Parameters::Parameters() {}

RuntimeError Parameters::Configure (std::shared_ptr < IEngine > in_engine,
                                    std::shared_ptr < IModel > in_model) {
  RuntimeError error;
  return error;
}

std::shared_ptr < IEngine > Parameters::GetEngine () {
  return nullptr;
}

std::shared_ptr < IModel > Parameters::GetModel () {
  return nullptr;
}

RuntimeError Parameters::Get (const std::string &in_parameter, int &value) {
  RuntimeError error;
  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter, double &value) {
  RuntimeError error;
  return error;
}

RuntimeError Parameters::Get (const std::string &in_parameter,
                              std::string &value) {
  RuntimeError error;
  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              const std::string &in_value) {
  RuntimeError error;
  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              int in_value) {
  RuntimeError error;
  return error;
}

RuntimeError Parameters::Set (const std::string &in_parameter,
                              double in_value) {
  RuntimeError error;
  return error;
}

RuntimeError Parameters::List (std::vector < ParameterMeta > &metas) {
  RuntimeError error;
  return error;
}

}  // namespace edgetpu
}  // namespace r2i