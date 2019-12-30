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

#include "r2i/tflite/loader.h"

#include <fstream>
#include <tensorflow/lite/model.h>

#include "r2i/imodel.h"
#include "r2i/tflite/model.h"

namespace r2i {
namespace tflite {

std::shared_ptr<r2i::IModel> Loader::Load (const std::string &in_path,
    r2i::RuntimeError &error) {

  std::ifstream graphdef_file;

  if (in_path.empty()) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE, "Received empty path to file");
    return nullptr;
  }

  graphdef_file.open (in_path, std::ios::binary | std::ios::ate);
  if (false == graphdef_file.is_open()) {
    error.Set (RuntimeError::Code::FILE_ERROR, "Unable to open file");
    return nullptr;
  }

  graphdef_file.close ();

  std::unique_ptr<::tflite::FlatBufferModel> tflite_model;
  ::tflite::ErrorReporter *error_reporter = ::tflite::DefaultErrorReporter();
  tflite_model = ::tflite::FlatBufferModel::BuildFromFile(in_path.c_str(),
                 error_reporter);

  delete error_reporter;

  if (nullptr == tflite_model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "Unable to load TensorFlow Lite model");
    return nullptr;
  }

  std::shared_ptr<::tflite::FlatBufferModel> tflite_model_shared{std::move(tflite_model)};
  auto model = std::make_shared<Model>();

  error = model->Set(tflite_model_shared);

  if (error.IsError ()) {
    model = nullptr;
  }

  return model;
}
}
}
