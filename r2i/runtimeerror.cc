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

#include "r2i/runtimeerror.h"

namespace r2i {
const std::string ok_string = "Everything went OK";

RuntimeError::RuntimeError ()
  : description(ok_string), code(Code::EOK) {
}

RuntimeError::RuntimeError (Code code, const std::string &description)
  : description(description), code(code) {
}

void RuntimeError::Clean () {
  this->Set (Code::EOK, ok_string);
}

void RuntimeError::Set (Code code, const std::string &description) {
  this->code = code;
  this->description = description;
}

const std::string RuntimeError::GetDescription () {
  return this->description;
}

RuntimeError::Code RuntimeError::GetCode () {
  return this->code;
}

}
