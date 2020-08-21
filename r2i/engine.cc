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

#include "engine.h"

namespace r2i {

RuntimeError Engine::SetModel (std::shared_ptr<r2i::IModel> in_model) {
  RuntimeError error;
  error.Set(RuntimeError::NOT_IMPLEMENTED, "Engine::SetModel method not implemented");
  return error;
}

RuntimeError Engine::Start () {
  RuntimeError error;
  error.Set(RuntimeError::NOT_IMPLEMENTED, "Engine::Start method not implemented");
  return error;
}

RuntimeError Engine::Stop () {
  RuntimeError error;
  error.Set(RuntimeError::NOT_IMPLEMENTED, "Engine::Stop method not implemented");
  return error;
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
      in_frame, r2i::RuntimeError &error) {
  error.Set(RuntimeError::NOT_IMPLEMENTED, "Engine::Predict method not implemented");
  return nullptr;
}

RuntimeError Engine::SetPreprocessing (std::shared_ptr<IPreprocessing> preprocessing) {
  RuntimeError error;

  if (nullptr == preprocessing) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Trying to set engine preprocessing with null pointer");
    return error;
  }

  this->preprocessing = preprocessing;

  return error;
}

std::shared_ptr<IPreprocessing> Engine::GetPreprocessing () {
  return this->preprocessing;
}

RuntimeError Engine::SetPostprocessing (std::shared_ptr<IPostprocessing> postprocessing) {
  RuntimeError error;

  if (nullptr == postprocessing) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Trying to set engine postprocessing with null pointer");
    return error;
  }

  this->postprocessing = postprocessing;

  return error;
}

std::shared_ptr<IPostprocessing> Engine::GetPostprocessing () {
  return this->postprocessing;
}

}
