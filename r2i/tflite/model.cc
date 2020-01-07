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

#include "r2i/tflite/model.h"

namespace r2i {
namespace tflite {

Model::Model () {
  this->tflite_model = nullptr;
}

RuntimeError Model::Start (const std::string &name) {
  RuntimeError error;

  return error;
}

RuntimeError Model::Set (std::shared_ptr<::tflite::FlatBufferModel> tfltmodel) {
  RuntimeError error;

  if (nullptr != tfltmodel) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Trying to set model with null model pointer");
    return error;
  }

  this->tflite_model = tfltmodel;

  return error;
}

} // namespace tflite
} // namespace r2i
