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

void RuntimeError::Clean () {
  this->Set (Code::EOK, "Everything OK");
}

void RuntimeError::Set (Code code, const std::string &description) {
  this->code = code;
  this->description = description;
}

}
