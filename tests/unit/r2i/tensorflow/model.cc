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

#include <memory>
#include <r2i/r2i.h>
#include <r2i/tensorflow/model.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

bool invalid_input_node;
bool invalid_output_node;

void TF_GraphImportGraphDef(TF_Graph *graph, const TF_Buffer *graph_def,
                            const TF_ImportGraphDefOptions *options, TF_Status *status) {}
void TF_DeleteGraph(TF_Graph *g) { return; }
TF_Graph *TF_NewGraph() { return nullptr; }
TF_Operation *TF_GraphOperationByName(TF_Graph *graph, const char *oper_name) {
  if (invalid_input_node || invalid_output_node) {
    return nullptr;
  } else {
    return (TF_Operation *) oper_name;
  }
}

TEST_GROUP (TensorflowModel) {
  r2i::RuntimeError error;
  r2i::tensorflow::Model model;
  std::shared_ptr<TF_Buffer> buffer;

  void setup () {
    buffer = std::shared_ptr<TF_Buffer> (nullptr);
    model.SetInputLayerName("input-value");
    model.SetOutputLayerName("output-value");
    invalid_input_node = false;
    invalid_output_node = false;
  }

  void teardown () {
  }
};

TEST (TensorflowModel, Start) {
  error = model.Start ("graph");
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowModel, StartEmptyInputName) {
  model.SetInputLayerName("");
  error = model.Start ("");
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorflowModel, StartEmptyOutputName) {
  model.SetOutputLayerName("");
  error = model.Start ("");
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorflowModel, StartInvalidInputNode) {
  invalid_input_node = true;
  error = model.Start ("");
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorflowModel, StartInvalidOutputNode) {
  invalid_output_node = true;
  error = model.Start ("");
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorflowModel, SetAndGetInputLayerName) {
  std::string in_value;
  std::string out_value;

  in_value = "myinputlayer";
  error = model.SetInputLayerName(in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  out_value = model.GetInputLayerName();

  STRCMP_EQUAL(in_value.c_str(), out_value.c_str());
}

TEST (TensorflowModel, SetAndGetOutputLayerName) {
  std::string in_value;
  std::string out_value;

  in_value = "myoutputlayer";
  error = model.SetOutputLayerName(in_value);

  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  out_value = model.GetOutputLayerName();

  STRCMP_EQUAL(in_value.c_str(), out_value.c_str());
}

TEST (TensorflowModel, LoadNullBuffer) {
  error = model.Load (nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowModel, LoadSuccess) {
  error = model.Load (buffer);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowModel, SetMultipleOutputs) {
  std::string output_1 = "output-value-1";
  std::string output_2 = "output-value-2";
  std::string output_3 = "output-value-3";
  std::vector< std::string > output_layers;

  output_layers.push_back(output_1);
  output_layers.push_back(output_2);
  output_layers.push_back(output_3);

  error = model.SetOutputLayersNames(output_layers);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  error = model.Start("");
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowModel, CheckMultipleOutputs) {
  std::string output_1 = "output-value-1";
  std::string output_2 = "output-value-2";
  std::string output_3 = "output-value-3";
  std::vector< std::string > output_layers;
  std::vector< std::string > output_layers_test;

  output_layers.push_back(output_1);
  output_layers.push_back(output_2);
  output_layers.push_back(output_3);

  error = model.SetOutputLayersNames(output_layers);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  output_layers_test = model.GetOutputLayesrNames();
  LONGS_EQUAL (output_layers.size(), output_layers_test.size());
  STRCMP_EQUAL(output_layers_test[0].c_str(), output_1.c_str());
  STRCMP_EQUAL(output_layers_test[1].c_str(), output_2.c_str());
  STRCMP_EQUAL(output_layers_test[2].c_str(), output_3.c_str());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
