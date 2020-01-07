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

#ifndef R2I_TFLITE_MODEL_H
#define R2I_TFLITE_MODEL_H

#include <memory>

#include <r2i/imodel.h>
#include <r2i/runtimeerror.h>
#include <tensorflow/lite/model.h>

namespace r2i {
namespace tflite {

class Model : public IModel {
 public:
  Model ();

  RuntimeError Start (const std::string &name) override;

  RuntimeError Set (std::shared_ptr<::tflite::FlatBufferModel> tfltmodel);

 private:
  std::shared_ptr<::tflite::FlatBufferModel> tflite_model;
};

}
}

#endif //R2I_TFLITE_MODEL_H
