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
#include <r2i/tensorflow/prediction.h>
#include <fstream>
#include <cstring>
#include <memory>

#include <tensorflow/c/c_api.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/TestHarness.h>

#define INPUTS 3

bool model_sucess = true;

int TF_GraphGetTensorNumDims(TF_Graph *graph, TF_Output output,
                             TF_Status *status) {
  return 1;
}
static void DummyDeallocator (void *data, size_t len, void *arg) {
  //NOP
}
void TF_GraphGetTensorShape(TF_Graph *graph, TF_Output output, int64_t *dims,
                            int num_dims, TF_Status *status) {
  dims[0] = 3;
}
TF_DataType TF_OperationOutputType(TF_Output oper_out) {
  if (model_sucess) {
    return TF_FLOAT;
  } else {
    return TF_DOUBLE;
  }
}
TF_Graph *TF_NewGraph() { return (TF_Graph *) new int; }
void TF_DeleteGraph(TF_Graph *g) {
  if (g == nullptr) return;
  delete (int *) g;
}

TEST_GROUP (TensorflowPrediction) {
  r2i::RuntimeError error;

  r2i::tensorflow::Prediction prediction;
  float matrix[INPUTS] = {0.2, 0.4, 0.6};
  std::shared_ptr<TF_Graph> pgraph;
  TF_Operation *poperation;
  std::shared_ptr<TF_Tensor> pout_tensor;

  int64_t raw_input_dims[1] = {INPUTS};

  void setup () {
    pgraph = std::shared_ptr<TF_Graph> (TF_NewGraph (), TF_DeleteGraph);

    poperation = (TF_Operation *)
                 &matrix; /* This is only used to avoid nullptr error, the actual use of this function is captured by TF_OperationOutputType */

    pout_tensor = std::shared_ptr<TF_Tensor> (TF_NewTensor(TF_FLOAT, raw_input_dims,
                  1, matrix, INPUTS * sizeof(float), DummyDeallocator, NULL), TF_DeleteTensor);
  }

  void teardown () {
  }
};

TEST (TensorflowPrediction, SetTensorSuccess) {
  r2i::RuntimeError error;

  error = prediction.SetTensor(pout_tensor);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
}

TEST (TensorflowPrediction, SetTensorNullTensor) {
  r2i::RuntimeError error;

  error = prediction.SetTensor(nullptr);
  LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode());
}

TEST (TensorflowPrediction, SetTensorIncompatibleModel) {
  r2i::RuntimeError error;
  model_sucess = false;

  error = prediction.SetTensor(pout_tensor);
  LONGS_EQUAL (r2i::RuntimeError::Code::INCOMPATIBLE_MODEL, error.GetCode());

  model_sucess = true;
}

TEST (TensorflowPrediction, Prediction) {
  r2i::RuntimeError error;
  double result = 0;

  error = prediction.SetTensor(pout_tensor);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  result = prediction.At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());
  DOUBLES_EQUAL (matrix[0], result, 0.05);
}

TEST (TensorflowPrediction, PredictionNoTensor) {
  r2i::RuntimeError error;

  prediction.At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorflowPrediction, PredictionNoData) {
  r2i::RuntimeError error;

  pout_tensor = std::shared_ptr<TF_Tensor> (TF_NewTensor(TF_FLOAT, raw_input_dims,
                1, nullptr, INPUTS * sizeof(float), DummyDeallocator, NULL), TF_DeleteTensor);

  error = prediction.SetTensor(pout_tensor);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  prediction.At (0, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               error.GetCode());
}

TEST (TensorflowPrediction, PredictionNonExistentIndex) {
  r2i::RuntimeError error;

  error = prediction.SetTensor(pout_tensor);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode());

  prediction.At (5, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::MEMORY_ERROR, error.GetCode());
}

int main (int ac, char **av) {
  /* This module detects fake leaks since the TF_Tensor couldn't be mocked since it's directly used by the predict module */
  MemoryLeakWarningPlugin::turnOffNewDeleteOverloads();

  return CommandLineTestRunner::RunAllTests (ac, av);
}
