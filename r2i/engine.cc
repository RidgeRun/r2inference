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
  error.Set(RuntimeError::NOT_IMPLEMENTED,
            "Engine::SetModel method not implemented");
  return error;
}

RuntimeError Engine::Start () {
  RuntimeError error;
  error.Set(RuntimeError::NOT_IMPLEMENTED,
            "Engine::Start method not implemented");
  return error;
}

RuntimeError Engine::Stop () {
  RuntimeError error;
  error.Set(RuntimeError::NOT_IMPLEMENTED, "Engine::Stop method not implemented");
  return error;
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) {
  /* Apply preprocessing, if any */
  error =  Preprocess (*in_frame);
  if (error.IsError ()) {
    return nullptr;
  }
  auto prediction = Process(in_frame, error);
  if (error.IsError ()) {
    return nullptr;
  }
  /* Apply postprocessing, if any */
  error =  Postprocess (*prediction);
  if (error.IsError ()) {
    return nullptr;
  }
  return prediction;
}

RuntimeError Engine::SetPreprocessing (std::shared_ptr<IPreprocessing>
                                       preprocessing) {
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

RuntimeError Engine::SetPostprocessing (std::shared_ptr<IPostprocessing>
                                        postprocessing) {
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

RuntimeError Engine::Preprocess (IFrame &data) {
  RuntimeError error;

  /* No preprocessing module set, don't do preprocessing then */
  if (nullptr == this->preprocessing) {
    error.Set(RuntimeError::Code::EOK,
              "Preprocessing module has not been set");
    return error;
  }

  error = this->preprocessing->Apply(data);

  return error;
}

RuntimeError Engine::Postprocess (IPrediction &prediction) {
  RuntimeError error;

  /* No postprocessing module set, don't do postprocessing then */
  if (nullptr == this->postprocessing) {
    error.Set(RuntimeError::Code::EOK,
              "Postprocessing module has not been set");
    return error;
  }

  error = this->postprocessing->Apply(prediction);

  return error;
}

std::shared_ptr<r2i::IPrediction> Engine::Process (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) {
  error.Set(RuntimeError::NOT_IMPLEMENTED,
            "Engine::Process method not implemented");
  return nullptr;
}

}
