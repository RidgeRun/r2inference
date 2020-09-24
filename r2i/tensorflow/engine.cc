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
#include "r2i/tensorflow/frame.h"
#include "r2i/tensorflow/prediction.h"

namespace r2i {
namespace tensorflow {

#define RAM_ARRAY_DEFAULT_INDEX -1
#define RAM_ARRAY_SIZE 11

const uint8_t _gpu_mem_config[10][RAM_ARRAY_SIZE] = {
  {0x32, 0x9, 0x9, 0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xb9, 0x3f},
  {0x32, 0x9, 0x9, 0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xc9, 0x3f},
  {0x32, 0x9, 0x9, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xd3, 0x3f},
  {0x32, 0x9, 0x9, 0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xd9, 0x3f},
  {0x32, 0x9, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xe0, 0x3f},
  {0x32, 0x9, 0x9, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xe3, 0x3f},
  {0x32, 0x9, 0x9, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xe6, 0x3f},
  {0x32, 0x9, 0x9, 0x9a, 0x99, 0x99, 0x99, 0x99, 0x99, 0xe9, 0x3f},
  {0x32, 0x9, 0x9, 0xcd, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xec, 0x3f},
  {0x32, 0x9, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf0, 0x3f}
};

Engine::Engine () : state(State::STOPPED),
  session_memory_usage_index(RAM_ARRAY_DEFAULT_INDEX),
  session(nullptr), model(nullptr) {
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

RuntimeError Engine::SetMemoryUsage (double memory_usage) {
  RuntimeError error;

  if (memory_usage > 1.0 || memory_usage < 0.1) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE, "Invalid memory usage value");
    return error;
  }

  this->session_memory_usage_index = (static_cast<int>(memory_usage * 10) - 1);
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

  /* If the user didn't set a value then fallback to the
     installation's default */
  if (RAM_ARRAY_DEFAULT_INDEX != this->session_memory_usage_index) {
    TF_SetConfig(opt, _gpu_mem_config[this->session_memory_usage_index],
                 RAM_ARRAY_SIZE, status);
  }

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
  prediction->SetTensor (pout_tensor);

  return prediction;
}

RuntimeError Engine::Predict (std::shared_ptr<r2i::IFrame> in_frame,
                              std::vector< std::shared_ptr<r2i::IPrediction> > &predictions) {
  RuntimeError error;

  if (State::STARTED != this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine not started");
    return error;
  }

  /* These pointers are validated during load */
  auto pgraph = this->model->GetGraph ();
  auto in_operation = this->model->GetInputOperation ();
  std::vector< TF_Operation * > out_operations =
    this->model->GetOutputOperations();

  auto frame = std::dynamic_pointer_cast<Frame, IFrame> (in_frame);
  if (nullptr == frame) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided frame is not an tensorflow frame");
    return error;
  }

  auto pin_tensor = frame->GetTensor (pgraph, in_operation, error);
  if (error.IsError ()) {
    return error;
  }

  std::shared_ptr<TF_Status> pstatus(TF_NewStatus(), TF_DeleteStatus);

  auto *session = this->session.get ();
  auto *in_tensor = pin_tensor.get ();
  auto *status = pstatus.get ();

  std::vector<TF_Output> run_outputs;
  for (size_t index = 0; index < out_operations.size(); index++) {
    TF_Output run_output = {.oper = out_operations[index], .index = 0};

    run_outputs.push_back(run_output);
  }
  std::vector<TF_Tensor *> out_tensors (run_outputs.size());

  TF_Output run_inputs = {.oper = in_operation, .index = 0};

  TF_SessionRun(session,
                NULL,                                                 /* RunOptions */
                &run_inputs, &in_tensor, 1,                           /* Input tensors */
                &run_outputs[0], &out_tensors[0], out_tensors.size(), /* Output tensors */
                NULL, 0,                                              /* Target operations */
                NULL,                                                 /* RunMetadata */
                status);
  if (TF_GetCode(status) != TF_OK) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status));
    return error;
  }

  // Iterate over the multiple outputs
  for (auto &tensor: out_tensors) {
    std::shared_ptr<TF_Tensor> pout_tensor (tensor, TF_DeleteTensor);
    auto prediction = std::make_shared<Prediction>();

    prediction->SetTensor(pout_tensor);
    predictions.push_back(prediction);
  }

  return error;
}

Engine::~Engine () {
  this->Stop();
}

} //namespace tensorflow
} //namepsace r2i
