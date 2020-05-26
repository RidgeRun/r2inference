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

#ifndef R2I_TFLITE_ENGINE_H
#define R2I_TFLITE_ENGINE_H

#include <r2i/iengine.h>

#include <memory>

#include <r2i/tflite/model.h>
#include <tensorflow/lite/kernels/register.h>

namespace r2i {
namespace tflite {

class Engine : public IEngine {
 public:
  Engine ();

  r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) override;

  r2i::RuntimeError Start () override;

  r2i::RuntimeError Stop () override;

  std::shared_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
      in_frame, r2i::RuntimeError &error) override;
  RuntimeError SetNumberOfThreads (int number_of_threads);
  const int GetNumberOfThreads ();
  RuntimeError SetAllowFP16 (int allow_fp16);
  const int GetAllowFP16 ();
  int64_t GetRequiredBufferSize (TfLiteIntArray *dims);

  ~Engine ();

 protected:
  enum State {
    STARTED,
    STOPPED
  };

  State state;
  std::shared_ptr<::tflite::Interpreter> interpreter;
  std::shared_ptr<Model> model;
  int number_of_threads;
  int allow_fp16;

  virtual void SetupResolver(::tflite::ops::builtin::BuiltinOpResolver &resolver);
  virtual void SetInterpreterContext(std::shared_ptr<::tflite::Interpreter>
                                     interpreter);

 private:
  void PreprocessInputData(const float *input_data, const int size,
                           r2i::RuntimeError &error);
  float *GetOutputTensorData(r2i::RuntimeError &error);
  uint8_t ConvertToFixedPoint(const TfLiteTensor *tensor, float value);
  float ConvertToFloatingPoint(const TfLiteTensor *tensor, uint8_t value);

};

}
}
#endif //R2I_TFLITE_ENGINE_H
