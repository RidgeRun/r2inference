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

namespace r2i {
namespace tensorrt {

Engine::Engine () : state(State::STOPPED), model(nullptr) {
}

RuntimeError Engine::SetModel (std::shared_ptr<r2i::IModel> in_model) {

  RuntimeError error;

  if (State::STOPPED != this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Stop model before setting a new state");
    return error;
  }

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
  RuntimeError error;

  if (nullptr == this->model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Model not set yet");
    return error;
  }

  // error = this->model->Start("Tensorrt");
  // if (error.IsError ()) {
  //   return error;
  // }

  return error;
}

RuntimeError Engine::Stop () {
  RuntimeError error;

  // this->state = State::STOPPED;

  return error;
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) {
  ImageFormat in_format;

  error.Clean ();

  auto prediction = std::make_shared<Prediction>();
  // std::shared_ptr<TF_Status> pstatus(TF_NewStatus(), TF_DeleteStatus);

  // auto *session = this->session.get ();
  // auto *in_tensor = pin_tensor.get ();
  // auto *status = pstatus.get ();

  // TF_Output run_outputs = {.oper = out_operation, .index = 0};
  // TF_Output run_inputs = {.oper = in_operation, .index = 0};
  // TF_Tensor *out_tensor = nullptr;

  // TF_SessionRun(session,
  //               NULL,                         /* RunOptions */
  //               &run_inputs, &in_tensor, 1,   /* Input tensors */
  //               &run_outputs, &out_tensor, 1, /* Output tensors */
  //               NULL, 0,                      /* Target operations */
  //               NULL,                         /* RunMetadata */
  //               status);
  // if (TF_GetCode(status) != TF_OK) {
  //   error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
  //   return nullptr;
  // }

  // std::shared_ptr<TF_Tensor> pout_tensor (out_tensor, TF_DeleteTensor);
  // prediction->SetTensor (pgraph, out_operation, pout_tensor);

  return prediction;
}

Engine::~Engine () {
  this->Stop();
}

} //namespace tensorrt
} //namepsace r2i
