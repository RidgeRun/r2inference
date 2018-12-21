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

#include "r2i/tensorflow/engine.h"

#include <tensorflow/c/c_api.h>

#include "r2i/tensorflow/prediction.h"
#include "r2i/tensorflow/frame.h"

namespace r2i {
namespace tensorflow {

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
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null model");
    return error;
  }
  auto model = std::dynamic_pointer_cast<r2i::tensorflow::Model, r2i::IModel>
               (in_model);

  if (nullptr == model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided model is not an TENSORFLOW model");
    return error;
  }

  if (nullptr != this->model) {
    this->model = nullptr;
  }

  this->model = model;

  return error;
}

static RuntimeError FreeSession (TF_Session *session) {
  RuntimeError error;
  std::shared_ptr<TF_Status> pstatus(TF_NewStatus (), TF_DeleteStatus);
  TF_Status *status = pstatus.get();

  TF_CloseSession (session, status);
  if (TF_OK != TF_GetCode(status)) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
    return error;
  }
  TF_DeleteSession (session, status);
  if (TF_OK != TF_GetCode(status)) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
    return error;
  }

  return error;
}

RuntimeError Engine::Start ()  {
  RuntimeError error;

  if (State::STARTED == this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine already started");
    return error;
  }

  if (nullptr == this->model) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Model not set yet");
    return error;
  }

  std::shared_ptr<TF_Graph> pgraph = this->model->GetGraph ();
  std::shared_ptr<TF_Status> pstatus (TF_NewStatus (), TF_DeleteStatus);
  std::shared_ptr<TF_SessionOptions> popt (TF_NewSessionOptions(),
      TF_DeleteSessionOptions);

  TF_Graph *graph = pgraph.get();
  TF_Status *status = pstatus.get ();
  TF_SessionOptions *opt = popt.get ();

  std::shared_ptr<TF_Session> session (TF_NewSession(graph, opt, status),
                                       FreeSession);
  if (TF_GetCode(status) != TF_OK) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message(status));
    return error;
  }

  error = this->model->Start("Tensorflow");
  if (error.IsError ()) {
    return error;
  }

  this->session = session;
  this->state = State::STARTED;

  return error;
}

RuntimeError Engine::Stop () {
  RuntimeError error;

  if (State::STOPPED == this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine already stopped");
  }

  this->state = State::STOPPED;

  return error;
}

std::shared_ptr<r2i::IPrediction> Engine::Predict (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) {
  ImageFormat in_format;

  error.Clean ();

  if (State::STARTED != this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine not started");
    return nullptr;
  }

  /* These pointers are validated during load */
  auto pgraph = this->model->GetGraph ();
  auto out_operation = this->model->GetOutputOperation ();
  auto in_operation = this->model->GetInputOperation ();

  auto frame = std::dynamic_pointer_cast<Frame, IFrame> (in_frame);
  if (nullptr == frame) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided frame is not an tensorflow frame");
    return nullptr;
  }

  auto pin_tensor = frame->GetTensor (pgraph, in_operation, error);
  if (error.IsError ()) {
    return nullptr;
  }

  auto prediction = std::make_shared<Prediction>();
  std::shared_ptr<TF_Status> pstatus(TF_NewStatus(), TF_DeleteStatus);

  auto *session = this->session.get ();
  auto *in_tensor = pin_tensor.get ();
  auto *status = pstatus.get ();

  TF_Output run_outputs = {.oper = out_operation, .index = 0};
  TF_Output run_inputs = {.oper = in_operation, .index = 0};
  TF_Tensor *out_tensor = nullptr;

  TF_SessionRun(session,
                NULL,                         /* RunOptions */
                &run_inputs, &in_tensor, 1,   /* Input tensors */
                &run_outputs, &out_tensor, 1, /* Output tensors */
                NULL, 0,                      /* Target operations */
                NULL,                         /* RunMetadata */
                status);
  if (TF_GetCode(status) != TF_OK) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
    return nullptr;
  }

  std::shared_ptr<TF_Tensor> pout_tensor (out_tensor, TF_DeleteTensor);
  prediction->SetTensor (pgraph, out_operation, pout_tensor);

  return prediction;
}

Engine::~Engine () {
  this->Stop();
}

} //namespace tensorflow
} //namepsace r2i
