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

#include "r2i/iprediction.h"
#include "r2i/ncsdk/engine.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

RuntimeError Engine::SetModel (std::shared_ptr<r2i::IModel> in_model) {

  RuntimeError error;

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null model");
    return error;
  }
  auto model = std::dynamic_pointer_cast<r2i::ncsdk::Model, r2i::IModel>
               (in_model);

  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided engine is not an NCSDK engine");
    return error;
  }
  this->model = model;

  return error;
}

void Engine::Start (RuntimeError &error) {}

void Engine::Stop (RuntimeError &error) {}

std::unique_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame,
    r2i::RuntimeError &error) {
  return nullptr;

}

}
}

