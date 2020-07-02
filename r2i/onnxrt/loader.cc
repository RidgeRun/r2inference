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

#include "r2i/onnxrt/loader.h"

#include <fstream>
#include <memory>
#include <string>

#include "r2i/imodel.h"
#include "r2i/onnxrt/model.h"

namespace r2i {
namespace onnxrt {

std::shared_ptr<r2i::IModel> Loader::Load(const std::string &in_path,
    r2i::RuntimeError &error) {
  if (in_path.empty()) {
    error.Set(RuntimeError::Code::WRONG_API_USAGE,
              "Received NULL path to file");
    return nullptr;
  }

  std::ifstream graphdef_file;
  graphdef_file.open(in_path, std::ios::binary | std::ios::ate);
  if (false == graphdef_file.is_open()) {
    error.Set(RuntimeError::Code::FILE_ERROR, "Unable to open file");
    return nullptr;
  }

  size_t graphdef_size = graphdef_file.tellg();
  graphdef_file.seekg (0, std::ios::beg);

  this->model_data = std::shared_ptr<char>(new char[graphdef_size],
                     std::default_delete<char[]>());
  if (nullptr == this->model_data) {
    error.Set (RuntimeError::Code::MEMORY_ERROR,
               "Can not allocate memory for graphdef");
    return nullptr;
  }

  graphdef_file.read (model_data.get(), graphdef_size);
  graphdef_file.close();

  auto model = std::make_shared<Model>();

  error = model->SetOnnxrtModel(std::static_pointer_cast<void>(this->model_data),
                                graphdef_size);

  if (error.IsError()) {
    return nullptr;
  }

  return model;
}

}  // namespace onnxrt
}  // namespace r2i
