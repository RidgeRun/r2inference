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

#include "r2i/tensorrt/loader.h"

#include <fstream>
#include <iostream>
#include <vector>

#include "r2i/imodel.h"
#include "r2i/tensorrt/model.h"


struct Logger: public
  nvinfer1::ILogger {
  void log (Severity severity, const char *msg) override {
    std::string tag;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        tag = "INTERNAL_ERROR";
        break;
      case
          Severity::kERROR:
        tag = "ERROR";
        break;
      case
          Severity::kWARNING:
        tag = "WARNING";
        break;
      case
          Severity::kINFO:
        tag = "INFO";
        break;
      case
          Severity::kVERBOSE:
        tag = "VERBOSE";
        break;
    }
  }
};

template <class T>
static void tensorRTIFaceDeleter (T *p) {
  if (p) {
    p->destroy ();
  }
}

namespace r2i {
namespace tensorrt {

std::shared_ptr<r2i::IModel> Loader::Load (const std::string &in_path,
    r2i::RuntimeError &error) {
  error.Clean();

  Logger logger;

  if (in_path.empty()) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE, "Received empty path to file");
    return nullptr;
  }

  std::shared_ptr < nvinfer1::IRuntime > infer =
    std::shared_ptr < nvinfer1::IRuntime > (nvinfer1::createInferRuntime (logger),
        tensorRTIFaceDeleter<nvinfer1::IRuntime>);
  if (!infer) {
    error.Set (RuntimeError::Code::FRAMEWORK_ERROR, "Unable to create runtime");
    return nullptr;
  }

  std::vector < char > cached;
  size_t size { 0};

  std::ifstream file
  (in_path, std::ios::binary);

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
    return nullptr;
  }

  std::shared_ptr < nvinfer1::ICudaEngine > engine = std::shared_ptr
      < nvinfer1::ICudaEngine >
      (infer->deserializeCudaEngine (cached.data (), size, nullptr),
       tensorRTIFaceDeleter<nvinfer1::ICudaEngine>);
  if (!engine) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "Unable to load cached engine");
    return nullptr;
  }

  std::shared_ptr < nvinfer1::IExecutionContext> context = std::shared_ptr
      < nvinfer1::IExecutionContext> (engine->createExecutionContext (),
                                      tensorRTIFaceDeleter<nvinfer1::IExecutionContext>);
  if (!context) {
    error.Set (RuntimeError::Code::INCOMPATIBLE_MODEL,
               "Unable to load cached engine");
    return nullptr;
  }

  std::shared_ptr<r2i::tensorrt::Model> model = std::make_shared<Model>();

  error = model->SetContext(context);
  if (error.IsError ()) {
    model = nullptr;
    return model;
  } else {
    error = model->SetCudaEngine(engine);
    if (error.IsError ()) {
      model = nullptr;
      return model;
    }
  }

  return model;
}
}
}

