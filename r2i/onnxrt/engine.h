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

#ifndef R2I_ONNXRT_ENGINE_H
#define R2I_ONNXRT_ENGINE_H

#include <r2i/iengine.h>

#include <memory>
#include <vector>

#include <r2i/onnxrt/frame.h>
#include <r2i/onnxrt/model.h>
#include <r2i/onnxrt/prediction.h>

namespace r2i {
namespace onnxrt {

class Engine : public IEngine {
 public:
  Engine ();
  ~Engine ();

  r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) override;

  r2i::RuntimeError Start () override;

  r2i::RuntimeError Stop () override;

  std::shared_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
      in_frame, r2i::RuntimeError &error) override;

 protected:
  enum State {
    STARTED,
    STOPPED
  };

  State state;
  std::shared_ptr<Model> model;

 private:
  RuntimeError ValidateInputTensorShape (int channels, int height, int width,
                                         std::vector<int64_t> input_dims);
  RuntimeError ScoreModel (std::shared_ptr<Ort::Session> session,
                           std::shared_ptr<Frame> frame,
                           size_t input_image_size,
                           std::vector<int64_t> input_node_dims,
                           std::shared_ptr<Prediction> prediction);
};

}  // namespace onnxrt
}  // namespace r2i

#endif  //R2I_ONNXRT_ENGINE_H
