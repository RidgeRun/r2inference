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
#define EXPECTED_OUT_TENSOR_VAL 0

static bool preprocessing_fail = false;
static bool postprocessing_fail = false;
static bool null_input_name = false;
static bool invalid_input_number = false;
static bool get_input_number_fail = false;
static bool invalid_frame = false;
static bool session_create_fail = false;
static bool session_run_fail = false;
static bool session_null = false;
static bool session_create_null_env = false;

std::string dummy_string_input = "input";
std::string dummy_string_output = "output";
std::vector<float> dummy_output_tensor_values(OUTPUT_SIZE);
std::vector<float> dummy_frame_data(CHANNELS *FRAME_WIDTH *FRAME_HEIGHT);

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

/* Mocks for Pre/Post processing modules */
namespace mock {
class Preprocessing : public r2i::IPreprocessing {
 public:
  Preprocessing () {}
  r2i::RuntimeError apply(r2i::IFrame &data) {
    r2i::RuntimeError error;
    if (preprocessing_fail) {
      error.Set (r2i::RuntimeError::Code::NOT_IMPLEMENTED,
                 "Functionality not implement");
    }
    return error;
  }
  std::vector<r2i::ImageFormat> getAvailableFormats() {
    std::vector<r2i::ImageFormat> image_format;
    return image_format;
  }
  std::vector<std::vector<int>> getAvailableDataSizes() {
    std::vector<std::vector<int>> data_sizes;
    return data_sizes;
  }
};

class Postprocessing : public r2i::IPostprocessing {
 public:
  Postprocessing () {}
  r2i::RuntimeError apply(r2i::IPrediction &prediction) {
    r2i::RuntimeError error;
    if (postprocessing_fail) {
      error.Set (r2i::RuntimeError::Code::NOT_IMPLEMENTED,
                 "Functionality not implement");
    }
    return error;
  }
};
}

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
  return dummy_frame_data.data();
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

void Engine::CreateEnv() {

}

void Engine::CreateSessionOptions() {

}

void Engine::AppendSessionOptionsExecutionProvider(Ort::SessionOptions
    &session_options, r2i::RuntimeError &error) {

}

void Engine::CreateSession(const void *model_data,
                           size_t model_data_size,
                           RuntimeError &error) {
  if (session_create_null_env) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null Ort::Env pointer");
  }

  if (session_create_fail) {
    throw onnxrtexcep;
  }
}

size_t Engine::GetSessionInputCount(std::shared_ptr<Ort::Session> session,
                                    RuntimeError &error) {
  if (session_null) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null session pointer");
    return 0;
  }
  if (invalid_input_number) {
    return INVALID_INPUT_NUMBER;
  }
  if (get_input_number_fail) {
    throw onnxrtexcep;
  }
  return INPUT_NUMBER;
}

size_t Engine::GetSessionOutputCount(std::shared_ptr<Ort::Session> session,
                                     RuntimeError &error) {
  if (session_null) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null session pointer");
    return 0;
  }
  return OUTPUT_NUMBER;
}

std::vector<int64_t> Engine::GetSessionInputNodeDims(
  std::shared_ptr<Ort::Session> session, size_t index, RuntimeError &error) {
  std::vector<int64_t> vect{ BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS };
  if (session_null) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null session pointer");
  }
  return vect;
}

size_t Engine::GetSessionOutputSize(std::shared_ptr<Ort::Session> session,
                                    size_t index, RuntimeError &error) {
  if (session_null) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null session pointer");
    return 0;
  }
  return OUTPUT_SIZE;
}

const char *Engine::GetSessionInputName(std::shared_ptr<Ort::Session> session,
                                        size_t index, OrtAllocator *allocator, RuntimeError &error) {
  if (session_null) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null session pointer");
    return dummy_string_input.c_str();
  }
  if (null_input_name) {
    throw onnxrtexcep;
  }
  return dummy_string_input.c_str();
}

const char *Engine::GetSessionOutputName(std::shared_ptr<Ort::Session> session,
    size_t index, OrtAllocator *allocator, RuntimeError &error) {
  if (session_null) {
    error.Set (RuntimeError::Code:: NULL_PARAMETER,
               "Received null session pointer");
    return dummy_string_input.c_str();
  }
  if (null_input_name) {
    throw onnxrtexcep;
  }
  return dummy_string_output.c_str();
}

/* Mock for wrapper of Ort::Session::Run method */
float *Engine::SessionRun (std::shared_ptr<Ort::Session> session,
                           std::shared_ptr<Frame> frame,
                           size_t input_image_size,
                           std::vector<int64_t> input_node_dims,
                           std::vector<Ort::Value> &output_tensor,
                           RuntimeError &error) {
  if (session_run_fail) {
    throw onnxrtexcep;
  }

  /* Fill data with dummy values */
  for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    dummy_output_tensor_values[i] = (float)i / (OUTPUT_SIZE + 1);

  return dummy_output_tensor_values.data();
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

TEST (OnnxrtEngine, NullInputNameException) {
  r2i::RuntimeError error;
  null_input_name = true;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (OnnxrtEngine, CreateSessionNullEnvError) {
  r2i::RuntimeError error;
  session_create_null_env = true;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (OnnxrtEngine, CreateSessionException) {
  r2i::RuntimeError error;
  session_create_fail = true;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR, error.GetCode ());
}

TEST (OnnxrtEngine, GetInfoSessionNull) {
  r2i::RuntimeError error;
  session_null = true;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
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

TEST (OnnxrtEngine, StopStopEngine) {
  r2i::RuntimeError error;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Stop ();
  error = engine->Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (OnnxrtEngine, GetInputNumberException) {
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

TEST (OnnxrtEngine, EngineNotStarted) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine->Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE,
               error.GetCode ());
}

TEST (OnnxrtEngine, EnginePredictNullFrame) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;
  std::shared_ptr<r2i::IFrame> null_frame;

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine->Predict (null_frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::FRAMEWORK_ERROR,
               error.GetCode ());
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

TEST (OnnxrtEngine, EnginePredictPreprocessFail) {
  preprocessing_fail = true;
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;
  std::shared_ptr<r2i::IPreprocessing> preprocessing =
    std::make_shared<mock::Preprocessing>();

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->SetPreprocessing(preprocessing);

  prediction = engine->Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NOT_IMPLEMENTED,
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

  auto prediction_onnxrt =
    std::dynamic_pointer_cast<r2i::onnxrt::Prediction, r2i::IPrediction>
    (prediction);

  LONGS_EQUAL (prediction_onnxrt->At(0, error), EXPECTED_OUT_TENSOR_VAL);
}

TEST (OnnxrtEngine, EnginePredictPostprocessFail) {
  postprocessing_fail = true;
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;
  std::shared_ptr<r2i::IPostprocessing> postprocessing =
    std::make_shared<mock::Postprocessing>();

  error = engine->SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine->SetPostprocessing(postprocessing);

  prediction = engine->Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::NOT_IMPLEMENTED,
               error.GetCode ());
}

int main (int ac, char **av) {
  MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();
  return CommandLineTestRunner::RunAllTests (ac, av);
}
