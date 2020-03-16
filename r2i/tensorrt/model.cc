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

#include <fstream>
#include <iostream>
#include <vector>

#include "r2i/tensorrt/model.h"

namespace r2i {
namespace tensorrt {

void iRuntimeDeleter (nvinfer1::IRuntime *p) {
  if (p)
    p->destroy ();
}

void ICudaEngineDeleter (nvinfer1::ICudaEngine *p) {
  if (p)
    p->destroy ();
}

Model::Model () {
}

RuntimeError Model::Start (const std::string &name) {
  RuntimeError error;
  Logger logger;

  if (this->engine && this->infer) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Model can only be started twice if it failed");
    return error;
  }


  if (name.empty()) {
    error.Set (RuntimeError::Code::NULL_PARAMETER, "Received empty file name");
    return error;
  }

  this->infer =
    std::shared_ptr < nvinfer1::IRuntime > (nvinfer1::createInferRuntime (logger),
        iRuntimeDeleter);
  if (!this->infer) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, "Unable to create runtime");
    return error;
  }

  std::vector < char > cached;
  size_t size { 0};

  std::ifstream file
  (name, std::ios::binary);

  if (file.good ()) {
    file.seekg (0, file.end);
    size = file.tellg ();
    file.seekg (0, file.beg);
    cached.resize (size);
    file.read (cached.data (), size);
    file.close ();
  } else {
    error.Set
    (RuntimeError::Code::FILE_ERROR, "Unable to load engine");
    return error;
  }

  this->engine = std::shared_ptr < nvinfer1::ICudaEngine >
                 (infer->deserializeCudaEngine (cached.data (), size, nullptr),
                  ICudaEngineDeleter);
  if (!this->engine) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, "Unable to load cached engine");
    return error;
  }

  return error;
}

} // namespace tensorrt
} // namespace r2i
