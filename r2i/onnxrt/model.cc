/* Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun. All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include "r2i/onnxrt/model.h"

#include <string>
#include <vector>

namespace r2i {
namespace onnxrt {

Model::Model() {
  this->model_data = nullptr;
}

RuntimeError Model::Start(const std::string &name) {
  RuntimeError error;

  return error;
}

RuntimeError Model::SetOnnxrtModel(std::shared_ptr<void> model_data,
                                   size_t model_data_size) {
  RuntimeError error;

  if (nullptr == model_data) {
    error.Set(RuntimeError::Code::NULL_PARAMETER,
              "Trying to set model data with null pointer");
    return error;
  }

  this->model_data = model_data;
  this->model_data_size = model_data_size;

  return error;
}

std::shared_ptr<void> Model::GetOnnxrtModel() {
  return this->model_data;
}

size_t Model::GetOnnxrtModelSize() {
  return this->model_data_size;
}

}  // namespace onnxrt
}  // namespace r2i
