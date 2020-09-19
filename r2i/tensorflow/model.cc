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

#include <iostream>

namespace r2i {
namespace tensorflow {

Model::Model () {
  this->graph = nullptr;
  this->buffer = nullptr;
  this->in_operation = nullptr;
  this->out_operations.clear();
  this->input_layer_name.clear ();
  this->output_layers_names.clear();
}

RuntimeError Model::Start (const std::string &name) {
  RuntimeError error;

  if (this->input_layer_name.empty()) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Input layer name has not been set");
    return error;
  }

  if (this->output_layers_names.size() == 0) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Output layers names has not been set");
    return error;
  }

  TF_Graph *graph = this->graph.get();

  TF_Operation *in_operation = nullptr;
  in_operation = TF_GraphOperationByName(graph, this->input_layer_name.c_str ());
  if (nullptr == in_operation) {
    error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
               "No valid input node provided");
    return error;
  }

  for (size_t index = 0; index < this->output_layers_names.size(); index++) {
    std::string out_layer_name = this->output_layers_names[index];

    if (out_layer_name.empty()) {
      error.Set (RuntimeError::Code::NULL_PARAMETER, "Invalid output layer name");
      return error;
    }

    TF_Operation *out_operation = nullptr;
    out_operation = TF_GraphOperationByName(graph, out_layer_name.c_str ());

    if (nullptr == out_operation) {
      error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
                 "No valid output node provided");
      return error;
    }

    this->out_operations.push_back(out_operation);
  }

  this->in_operation = in_operation;

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

/*
 * NOTE: This method should be removed once the multiple output
 * support is integrated with the GStreamer plugins. The logic of
 * this method is emulating the previous one to allow backward
 * compatibility.
 */
TF_Operation *Model::GetOutputOperation () {
  if (this->out_operations.size() > 0) {
    return this->out_operations[0];
  }
  return nullptr;
}

std::vector<TF_Operation *> Model::GetOutputOperations () {
  return this->out_operations;
}

RuntimeError Model::SetInputLayerName (const std::string &name) {
  this->input_layer_name = name;

  return RuntimeError ();
}

/*
 * NOTE: This method should be removed once the multiple output
 * support is integrated with the GStreamer plugins. The logic of
 * this method is emulating the previous one to allow backward
 * compatibility.
 */
RuntimeError Model::SetOutputLayerName (const std::string &name) {
  this->output_layers_names.clear();
  this->output_layers_names.push_back(name);
  return RuntimeError ();
}

RuntimeError Model::SetOutputLayersNames (std::vector< std::string > names) {
  this->output_layers_names = names;
  return RuntimeError ();
}

const std::string Model::GetInputLayerName () {
  return this->input_layer_name;
}

/*
 * NOTE: This method should be removed once the multiple output
 * support is integrated with the GStreamer plugins. The logic of
 * this method is emulating the previous one to allow backward
 * compatibility.
 */
const std::string Model::GetOutputLayerName () {
  std::string output_name;

  if (this->output_layers_names.size() > 0) {
    output_name = this->output_layers_names[0];
  }
  return output_name;
}

std::vector< std::string > Model::GetOutputLayesrNames () {
  return this->output_layers_names;
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
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL, TF_Message (status));
    return error;
  }

  this->buffer = pbuffer;
  this->graph = pgraph;


  return error;
}

} // namespace tensorflow
} // namespace r2i
