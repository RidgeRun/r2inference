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

struct TensorInfo {
  int num_dims = 0;
  std::vector<int64_t> dims;
  TF_DataType type;
  size_t type_size = 0;
  size_t data_size = 0;
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
  std::vector< std::shared_ptr<r2i::IPrediction> > predictions;

  error = this->Predict (in_frame, predictions);

  if (predictions.size() > 0) {
    return predictions.at(0);
  } else {
    return nullptr;
  }
}

RuntimeError Engine::Predict (std::shared_ptr<r2i::IFrame> in_frame,
                              std::vector< std::shared_ptr<r2i::IPrediction> > &predictions) {
  RuntimeError error;

  if (State::STARTED != this->state) {
    error.Set (RuntimeError::Code::WRONG_ENGINE_STATE,
               "Engine not started");
    return error;
  }

  auto frame = std::dynamic_pointer_cast<Frame, IFrame> (in_frame);
  if (nullptr == frame) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "The provided frame is not an tensorflow frame");
    return error;
  }

  /* These pointers are validated during load */
  auto graph = this->model->GetGraph ();

  std::vector<TF_Output> run_outputs = this->model->GetRunOutputs();
  std::vector<TF_Tensor *> out_tensors (run_outputs.size());

  std::vector<TF_Output> run_inputs = this->model->GetRunInputs();
  std::vector<TF_Tensor *> in_tensors;

  if (0 == run_inputs.size()) {
    error.Set(RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
              "No input layers provided");
    return error;
  }

  TF_Operation *in_operation = run_inputs.at(0).oper;
  auto in_tensor = frame->GetTensor (graph, in_operation, error);
  if (error.IsError ()) {
    return error;
  }
  in_tensors.push_back (in_tensor.get());

  std::shared_ptr<TF_Status> status(TF_NewStatus(), TF_DeleteStatus);

  TF_SessionRun(session.get(),
                NULL,                                                       /* RunOptions */
                run_inputs.data(), in_tensors.data(), in_tensors.size(),    /* Input tensors */
                run_outputs.data(), out_tensors.data(), out_tensors.size(), /* Output tensors */
                NULL, 0,                                                    /* Target operations */
                NULL,                                                       /* RunMetadata */
                status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, TF_Message (status.get()));
    return error;
  }

  // Iterate over the multiple outputs
  for (auto &tensor : out_tensors) {
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
