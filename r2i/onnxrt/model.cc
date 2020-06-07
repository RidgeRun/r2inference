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

#include "r2i/onnxrt/model.h"

#include <onnxruntime/core/common/exceptions.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <iostream>
#include <string>
#include <vector>

namespace r2i {
namespace onnxrt {

Model::Model() { this->session_ptr = nullptr; }

RuntimeError Model::Start(const std::string &name) {
  RuntimeError error;

  return error;
}

RuntimeError Model::Set(std::shared_ptr<Ort::Session> onnxrt_session) {
  RuntimeError error;

  if (nullptr == onnxrt_session) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Trying to set model session with null session pointer");
    return error;
  }

  this->session_ptr = onnxrt_session;

  return error;
}

std::shared_ptr<Ort::Session> Model::GetOnnxrtSession() {
  return this->session_ptr;
}

std::vector<const char *> Model::GetInputNodeNames(RuntimeError &error) {
  Ort::AllocatorWithDefaultOptions allocator;

  if (nullptr == this->session_ptr) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Trying to obtain input info from null session pointer");
    return this->input_node_names;
  }

  size_t num_input_nodes = this->session_ptr->GetInputCount();
  this->input_node_names.resize(num_input_nodes);

  for (unsigned int i = 0; i < num_input_nodes; i++) {
    char *input_name = this->session_ptr->GetInputName(i, allocator);
    this->input_node_names[i] = input_name;
  }

  return this->input_node_names;
}

std::vector<const char *> Model::GetOutputNodeNames(RuntimeError &error) {
  Ort::AllocatorWithDefaultOptions allocator;

  if (nullptr == this->session_ptr) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Trying to obtain output info from null session pointer");
    return this->output_node_names;
  }

  size_t num_output_nodes = this->session_ptr->GetOutputCount();
  this->output_node_names.resize(num_output_nodes);

  for (unsigned int i = 0; i < num_output_nodes; i++) {
    char *output_name = this->session_ptr->GetOutputName(i, allocator);
    this->output_node_names[i] = output_name;
  }

  return this->output_node_names;
}

}  // namespace onnxrt
}  // namespace r2i
