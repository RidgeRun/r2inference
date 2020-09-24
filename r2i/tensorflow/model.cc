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
  this->run_inputs.clear();
  this->run_outputs.clear();
  this->input_layer_name.clear ();
  this->output_layers_names.clear();
}

static RuntimeError FillRuns (std::shared_ptr<TF_Graph> graph,
			      const std::vector<std::string> &names,
			      std::vector<TF_Output> &runs) {
  RuntimeError error;

  for (auto &name: names) {

    if (name.empty()) {
      error.Set (RuntimeError::Code::NULL_PARAMETER, "Invalid layer name " + name);
      return error;
    }

    TF_Operation *operation = TF_GraphOperationByName(graph.get(), name.c_str ());
    if (nullptr == operation) {
      error.Set (RuntimeError::Code::INVALID_FRAMEWORK_PARAMETER,
                 "No valid node for layer " + name);
      return error;
    }

    runs.push_back({.oper = operation, .index = 0});
  }

  return error;
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

  error = FillRuns (this->graph, this->output_layers_names, this->run_outputs);
  if (error.IsError()) {
    return error;
  }

  error = FillRuns (this->graph, {this->input_layer_name}, this->run_inputs);
  if (error.IsError()) {
    return error;
  }

  return error;
}

std::shared_ptr<TF_Graph> Model::GetGraph () {
  return this->graph;
}

std::shared_ptr<TF_Buffer> Model::GetBuffer () {
  return this->buffer;
}

std::vector<TF_Output> Model::GetRunInputs () {
  return this->run_inputs;
}

std::vector<TF_Output> Model::GetRunOutputs () {
  return this->run_outputs;
}

RuntimeError Model::SetInputLayerName (const std::string &name) {
  this->input_layer_name = name;

  return RuntimeError ();
}

RuntimeError Model::SetOutputLayersNames (std::vector< std::string > names) {
  this->output_layers_names = names;
  return RuntimeError ();
}

const std::string Model::GetInputLayerName () {
  return this->input_layer_name;
}

std::vector< std::string > Model::GetOutputLayersNames () {
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
