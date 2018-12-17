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

#include "r2i/tensorflow/model.h"

namespace r2i {
namespace tensorflow {

Model::Model () {
  this->graph = nullptr;
  this->buffer = nullptr;
  this->out_operation = nullptr;
  this->in_operation = nullptr;
}

RuntimeError Model::Start (const std::string &name) {
  RuntimeError error;

  return error;
}

std::shared_ptr<TF_Graph> Model::GetGraph () {
  return this->graph;
}

std::shared_ptr<TF_Buffer> Model::GetBuffer () {
  return this->buffer;
}

TF_Operation *Model::GetInputOperation () {
  return this->in_operation;
}

TF_Operation *Model::GetOutputOperation () {
  return this->out_operation;
}

RuntimeError Model::Load (std::shared_ptr<TF_Buffer> pbuffer) {
  RuntimeError error;

  if (nullptr != buffer) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Trying to load model with null buffer");
    return error;
  }

  std::shared_ptr<TF_Status> pstatus (TF_NewStatus (), TF_DeleteStatus);
  std::shared_ptr<TF_Graph> pgraph (TF_NewGraph (), TF_DeleteGraph);
  std::shared_ptr<TF_ImportGraphDefOptions> pgopts (TF_NewImportGraphDefOptions(),
      TF_DeleteImportGraphDefOptions);

  TF_Buffer *buffer = pbuffer.get ();
  TF_Status *status = pstatus.get ();
  TF_Graph *graph = pgraph.get ();
  TF_ImportGraphDefOptions *gopts = pgopts.get ();

  TF_GraphImportGraphDef (graph, buffer, gopts, status);

  TF_Code code = TF_GetCode(status);
  if (code != TF_OK) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               GetStringFromStatus (code, error));
    return error;
  }

  TF_Operation *in_operation = nullptr;
  in_operation = TF_GraphOperationByName(graph, "input");
  if (nullptr == in_operation) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "No valid input node provided");
    return error;
  }

  TF_Operation *out_operation = nullptr;
  out_operation = TF_GraphOperationByName(graph,
                                          "InceptionV4/Logits/Predictions");
  if (nullptr == out_operation) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "No valid output node provided");
    return error;
  }

  this->buffer = pbuffer;
  this->graph = pgraph;
  this->in_operation = in_operation;
  this->out_operation = out_operation;

  return error;
}

} // namespace tensorflow
} // namespace r2i
