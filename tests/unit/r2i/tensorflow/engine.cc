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

#include <r2i/r2i.h>
#include <r2i/tensorflow/engine.h>
#include <r2i/tensorflow/frame.h>
#include <r2i/tensorflow/model.h>
#include <r2i/tensorflow/prediction.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

class MockModel : public r2i::IModel {
  r2i::RuntimeError Start (const std::string &name) override {r2i::RuntimeError error; return error;}
};

/* Mock implementation of actual classes*/
TF_Session *TF_NewSession(TF_Graph *graph, const TF_SessionOptions *opts,
                          TF_Status *status) { return nullptr; }
void TF_CloseSession(TF_Session *s, TF_Status *status) { return; }
void TF_DeleteSession(TF_Session *s, TF_Status *status) { return; }
void TF_SessionRun(TF_Session *session, const TF_Buffer *run_options,
                   const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs,
                   const TF_Output *outputs, TF_Tensor **output_values, int noutputs,
                   const TF_Operation *const *target_opers, int ntargets, TF_Buffer *run_metadata,
                   TF_Status *status) { }

namespace r2i {
namespace tensorflow {

Model::Model () {}
r2i::RuntimeError Model::Start (const std::string &name) { return RuntimeError(); }
r2i::RuntimeError Model::Load (std::shared_ptr<TF_Buffer> pbuffer) {
  this->buffer = pbuffer;
  return r2i::RuntimeError();
}
std::shared_ptr<TF_Graph> Model::GetGraph () {return this->graph;}
std::shared_ptr<TF_Buffer> Model::GetBuffer () {return this->buffer;}
TF_Operation *Model::GetInputOperation () { return nullptr; }
TF_Operation *Model::GetOutputOperation () { return nullptr; }
RuntimeError Model::SetInputLayerName (const std::string &name) { return RuntimeError(); }
RuntimeError Model::SetOutputLayerName (const std::string &name) { return RuntimeError(); }
const std::string Model::GetInputLayerName () { return "inputLayer"; }
const std::string Model::GetOutputLayerName () { return "outputLayer"; }

Frame::Frame () {}
RuntimeError Frame::Configure (void *in_data, int width, int height,
                               r2i::ImageFormat::Id format) { return RuntimeError(); }
void *Frame::GetData () { return nullptr; }
int Frame::GetWidth () { return 0; }
int Frame::GetHeight () { return 0; }
ImageFormat Frame::GetFormat () { return ImageFormat::Id::UNKNOWN_FORMAT; }
std::shared_ptr<TF_Tensor> Frame::GetTensor (std::shared_ptr<TF_Graph> graph,
    TF_Operation *operation, RuntimeError &error) { return nullptr; }

Prediction::Prediction () {}
double Prediction::At (unsigned int index,  r2i::RuntimeError &error) { return 0.0; }
void *Prediction::GetResultData () { return nullptr; }
unsigned int Prediction::GetResultSize () { return 0; }
RuntimeError Prediction::SetTensor (std::shared_ptr<TF_Graph> graph,
                                    TF_Operation *operation, std::shared_ptr<TF_Tensor> tensor) { return RuntimeError(); }
}
}

TEST_GROUP (TensorflowEngine) {
  r2i::tensorflow::Engine engine;
  std::shared_ptr<r2i::IModel> model;
  std::shared_ptr<r2i::IModel> inc_model;
  std::shared_ptr<r2i::IFrame> frame;

  void setup () {
    model = std::make_shared<r2i::tensorflow::Model> ();

    auto tf_model = std::dynamic_pointer_cast<r2i::tensorflow::Model, r2i::IModel>
                    (model);
    tf_model->SetInputLayerName("InputLayer");
    tf_model->SetOutputLayerName("OutputLayer");

    inc_model = std::make_shared<MockModel> ();
    frame = std::make_shared<r2i::tensorflow::Frame> ();
  }
};

TEST (TensorflowEngine, SetModel) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorflowEngine, SetModelNull) {
  r2i::RuntimeError error;

  error = engine.SetModel (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}
TEST (TensorflowEngine, SetModelInvalid) {
  r2i::RuntimeError error;

  error = engine.SetModel (inc_model);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode ());
}

TEST (TensorflowEngine, StartEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorflowEngine, StartEngineEmpty) {
  r2i::RuntimeError error;

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}

TEST (TensorflowEngine, StartEngineTwice) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());
}

TEST (TensorflowEngine, StartStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

}

TEST (TensorflowEngine, StartStopEngineTwice) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (TensorflowEngine, StopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());

}

TEST (TensorflowEngine, StopStopEngine) {
  r2i::RuntimeError error;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Stop ();
  error = engine.Stop ();
  LONGS_EQUAL (r2i::RuntimeError::Code::WRONG_ENGINE_STATE, error.GetCode ());

}

TEST (TensorflowEngine, PredictEngine) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IPrediction> prediction;

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  prediction = engine.Predict (frame, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

}

TEST (TensorflowEngine, MultiplePredictsEngine) {
  r2i::RuntimeError error;
  std::string output_1 = "output-value-1";
  std::string output_2 = "output-value-2";
  std::string output_3 = "output-value-3";
  std::vector< std::string > output_layers;
  std::vector<std::shared_ptr<r2i::IPrediction>> predictions;

  output_layers.push_back(output_1);
  output_layers.push_back(output_2);
  output_layers.push_back(output_3);

  auto tf_model = static_cast<r2i::tensorflow::Model *>(model.get());
  error = tf_model->SetOutputLayersNames(output_layers);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.SetModel (model);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Start ();
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  error = engine.Predict (frame, predictions);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
