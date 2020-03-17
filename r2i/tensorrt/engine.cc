/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#include "r2i/tensorrt/engine.h"
#include "r2i/tensorrt/prediction.h"
#include "r2i/tensorrt/frame.h"

#include <vector>

namespace r2i {
namespace tensorrt {

Engine::Engine () : model(nullptr) {
}

RuntimeError Engine::SetModel (std::shared_ptr<r2i::IModel> in_model) {

  RuntimeError error;

  if (nullptr == in_model) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Received null model");
    return error;
  }
  auto model = std::dynamic_pointer_cast<r2i::tensorrt::Model, r2i::IModel>
               (in_model);

  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided model is not an TENSORRT model");
    return error;
  }

  this->model = model;

  return error;
}

RuntimeError Engine::Start ()  {
  return RuntimeError();
}

RuntimeError Engine::Stop () {
  return RuntimeError();
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) {
  ImageFormat in_format;

  error.Clean ();

  auto prediction = std::make_shared<Prediction>();

  int batchSize = 1;

  std::vector < void *> buffers;
  /* FIXME Store these in the correct order */
  buffers.emplace_back (in_frame->GetData());
  buffers.emplace_back (prediction->GetResultData());

  bool status = this->model->GetTRContext()->execute (batchSize, buffers.data ());
  if (!status) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR,
               "Unable to run prediction on model");
    return nullptr;
  }

  return prediction;
}

Engine::~Engine () {
  this->Stop();
}

} //namespace tensorrt
} //namepsace r2i
