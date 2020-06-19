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

#include <core/session/onnxruntime_cxx_api.h>

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

 private:
  enum State {
    STARTED,
    STOPPED
  };

  State state;
  std::shared_ptr<Model> model;
  std::vector<int64_t> input_node_dims;
  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
  size_t output_size;
  size_t num_input_nodes;
  size_t num_output_nodes;

  size_t GetSessionInputCount(std::shared_ptr<Ort::Session> session);
  size_t GetSessionOutputCount(std::shared_ptr<Ort::Session> session);
  std::vector<int64_t> GetSessionInputNodeDims(std::shared_ptr<Ort::Session>
      session, size_t index);
  size_t GetSessionOutputSize(std::shared_ptr<Ort::Session> session,
                              size_t index);
  char *GetSessionInputName(std::shared_ptr<Ort::Session> session, size_t index,
                            OrtAllocator *allocator);
  char *GetSessionOutputName(std::shared_ptr<Ort::Session> session, size_t index,
                             OrtAllocator *allocator);
  float *SessionRun (std::shared_ptr<Ort::Session> session,
                     std::shared_ptr<Frame> frame,
                     size_t input_image_size,
                     std::vector<int64_t> input_node_dims,
                     Ort::Value &input_tensor,
                     std::vector<Ort::Value> &output_tensor,
                     RuntimeError &error);
  r2i::RuntimeError GetSessionInfo(std::shared_ptr<Ort::Session> session,
                                   size_t index);
  r2i::RuntimeError ValidateInputTensorShape (int channels, int height, int width,
      std::vector<int64_t> input_dims);
  r2i::RuntimeError ScoreModel (std::shared_ptr<Ort::Session> session,
                                std::shared_ptr<Frame> frame,
                                size_t input_size,
                                size_t output_size,
                                std::vector<int64_t> input_node_dims,
                                std::shared_ptr<Prediction> prediction);
};

}  // namespace onnxrt
}  // namespace r2i

#endif  //R2I_ONNXRT_ENGINE_H
