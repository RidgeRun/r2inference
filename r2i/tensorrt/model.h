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

#ifndef R2I_TENSORRT_MODEL_H
#define R2I_TENSORRT_MODEL_H

#include <iostream>
#include <memory>
#include <NvInfer.h>
#include "NvInferPlugin.h"

#include <r2i/imodel.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace tensorrt {

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

    std::cerr << "[RR]" << "[" << tag << "] " << std::string (msg) << std::endl;
  }
};


class Model : public IModel {
 public:
  Model ();

  //RuntimeError Start (const std::string &name) override;
  RuntimeError Start (const std::string &name) override;

  std::shared_ptr<nvinfer1::ICudaEngine> GetTREngineModel ();

  RuntimeError Set (std::shared_ptr<nvinfer1::ICudaEngine> tensorrtmodel);
 private:

  std::shared_ptr < nvinfer1::ICudaEngine > engine;
};

}
}

#endif //R2I_TENSORRT_MODEL_H
