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

#include <r2i/r2i.h>
#include <r2i/onnxrt/engine.h>
#include <r2i/onnxrt/frame.h>
#include <r2i/onnxrt/model.h>

#include <core/common/exceptions.h>
#include <core/session/onnxruntime_cxx_api.h>

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

#define FRAME_WIDTH  224
#define FRAME_HEIGHT  224
#define CHANNELS 3
#define INVALID_FRAME_WIDTH  64
#define INVALID_FRAME_HEIGHT  64
#define INVALID_CHANNELS 1
#define INPUT_NUMBER 1
#define OUTPUT_NUMBER 1
#define INVALID_INPUT_NUMBER 2
#define BATCH_SIZE 1
#define OUTPUT_SIZE 1000

void *dummy_char = nullptr;
void *dummy_frame_ptr = nullptr;
float *dummy_out_data_ptr = nullptr;
static bool null_input_name = false;
static bool invalid_input_number = false;
static bool get_input_number_fail = false;
static bool invalid_frame = false;
static bool session_run_fail = false;

/* To simulate exceptions thrown by onnxruntime API. Exceptions in this
 * API are derived from std::exception.
 */
class OnnxrtExcep : public std::exception {
  virtual const char *what() const throw() { return "ONNXRT exception thrown"; }
} onnxrtexcep;

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {
    r2i::RuntimeError error;
    return error;
  }
};

namespace r2i {

int ImageFormat::GetNumPlanes () {
  if (invalid_frame) {
    return INVALID_CHANNELS;
  }
  return CHANNELS;
}

namespace onnxrt {

Frame::Frame () {}
void *Frame::GetData () {
  return dummy_frame_ptr;
}

int Frame::GetWidth () {
  if (invalid_frame) {
    return INVALID_FRAME_WIDTH;
  }

  return FRAME_WIDTH;
}

int Frame::GetHeight () {
  if (invalid_frame) {
    return INVALID_FRAME_HEIGHT;
  }

  return FRAME_HEIGHT;
}

size_t Engine::GetSessionInputCount(std::shared_ptr<Ort::Session> session) {
  if (invalid_input_number) {
    return INVALID_INPUT_NUMBER;
  }
  if (get_input_number_fail) {
    throw onnxrtexcep;
    return 0;
  }
  return INPUT_NUMBER;
}

size_t Engine::GetSessionOutputCount(std::shared_ptr<Ort::Session> session) {
  return OUTPUT_NUMBER;
}

std::vector<int64_t> Engine::GetSessionInputNodeDims(
  std::shared_ptr<Ort::Session> session, size_t index) {
  std::vector<int64_t> vect{ BATCH_SIZE, CHANNELS, FRAME_HEIGHT, FRAME_WIDTH };
  return vect;
}

size_t Engine::GetSessionOutputSize(std::shared_ptr<Ort::Session> session,
                                    size_t index) {
  return OUTPUT_SIZE;
}

char *Engine::GetSessionInputName(std::shared_ptr<Ort::Session> session,
                                  size_t index, OrtAllocator *allocator) {
  if (null_input_name) {
    throw onnxrtexcep;
    return nullptr;
  }
  return (char *)dummy_char;
}

char *Engine::GetSessionOutputName(std::shared_ptr<Ort::Session> session,
                                   size_t index, OrtAllocator *allocator) {
  if (null_input_name) {
    throw onnxrtexcep;
    return nullptr;
  }
  return (char *)dummy_char;
}

/* Mock for wrapper of Ort::Session::Run method */
float *Engine::SessionRun (std::shared_ptr<Ort::Session> session,
                           std::shared_ptr<Frame> frame,
                           size_t input_image_size,
                           std::vector<int64_t> input_node_dims,
                           Ort::Value &input_tensor,
                           std::vector<Ort::Value> &output_tensor,
                           RuntimeError &error) {
  if (session_run_fail) {
    throw onnxrtexcep;
    return nullptr;
  }
  return dummy_out_data_ptr;
}

}
}

TEST_GROUP (OnnxrtEngine) {
  std::shared_ptr<r2i::onnxrt::Engine> engine;
  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::IModel> inc_model;
  std::shared_ptr<r2i::IFrame> frame;

  void setup () {
    model = std::make_shared<r2i::onnxrt::Model> ();
    inc_model = std::make_shared<MockModel> ();
    engine = std::make_shared<r2i::onnxrt::Engine> ();
    frame = std::make_shared<r2i::onnxrt::Frame> ();
    dummy_char = malloc(sizeof(char));
    dummy_frame_ptr = malloc(sizeof(float));
    dummy_out_data_ptr = new float [OUTPUT_SIZE * sizeof(float)];
  }
  void teardown () {
    free(dummy_char);
    dummy_char = nullptr;
    free(dummy_frame_ptr);
    dummy_frame_ptr = nullptr;
    delete[] dummy_out_data_ptr;
    dummy_out_data_ptr = nullptr;
  }
};

TEST (OnnxrtEngine, SetModel) {
  r2i::RuntimeError error;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (OnnxrtEngine, SetModelNull) {
  r2i::RuntimeError error;

  error = engine->SetModel (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (OnnxrtEngine, SetModelInvalid) {
  r2i::RuntimeError error;

  error = engine->SetModel (inc_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (OnnxrtEngine, StartEngineNullSession) {
  r2i::RuntimeError error;
  null_input_name = true;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (OnnxrtEngine, StartEngineEmpty) {
  r2i::RuntimeError error;

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (OnnxrtEngine, StartEngineTwice) {
  r2i::RuntimeError error;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtEngine, StartStopEngine) {
  r2i::RuntimeError error;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  error = engine->Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (OnnxrtEngine, StopEngine) {
  r2i::RuntimeError error;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtEngine, InvalidModel) {
  r2i::RuntimeError error;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtEngine, StopStopEngine) {
  r2i::RuntimeError error;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Stop ();
  error = engine->Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtEngine, GetInputNumber) {
  r2i::RuntimeError error;
  get_input_number_fail = true;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (OnnxrtEngine, InvalidInputNumber) {
  r2i::RuntimeError error;
  invalid_input_number = true;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (OnnxrtEngine, EnginePredictInvalidFrame) {
  r2i::RuntimeError error;
  invalid_frame = true;
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine->Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode ());
}

TEST (OnnxrtEngine, EngineSessionRunFail) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;
  session_run_fail = true;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine->Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (OnnxrtEngine, EnginePredictSuccess) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine->Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
  return CommandLineTestRunner::RunAllTests (ac, av);
}
