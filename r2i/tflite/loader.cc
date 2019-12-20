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
#include <memory>
#include <tensorflow/lite/c/c_api.h>

#include "r2i/imodel.h"
#include "r2i/tflite/model.h"

namespace r2i {
namespace tflite {

std::shared_ptr<r2i::IModel> Loader::Load (const std::string &in_path,
    r2i::RuntimeError &error) {

  if (in_path.empty()) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE, "Received NULL path to file");
    return nullptr;
  }

  std::ifstream graphdef_file;
  graphdef_file.open (in_path, std::ios::binary | std::ios::ate);
  if (false == graphdef_file.is_open()) {
    error.Set (RuntimeError::Code::FILE_ERROR, "Unable to open file");
    return nullptr;
  }

  std::shared_ptr<TfLiteModel> tflite_model(TfLiteModelCreateFromFile(
        in_path.c_str()), TfLiteModelDelete);

  if (nullptr == tflite_model) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "Unable to load TensorFlow Lite model");
    graphdef_file.close ();
    return nullptr;
  }

  auto model = std::make_shared<Model>();

  error = model->Set(tflite_model);

  if (error.IsError ()) {
    model = nullptr;
  }
  graphdef_file.close ();

  return model;
}
}
}
