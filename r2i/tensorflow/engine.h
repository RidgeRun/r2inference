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

#ifndef R2I_TENSORFLOW_ENGINE_H
#define R2I_TENSORFLOW_ENGINE_H

#include <r2i/iengine.h>

#include <memory>

#include <r2i/tensorflow/model.h>

namespace r2i {
namespace tensorflow {

struct TensorInfo {
  int num_dims = 0;
  std::vector<int64_t> dims;
  TF_DataType type;
  size_t type_size = 0;
  size_t data_size = 0;
};

class Engine : public IEngine {
 public:
  Engine ();

  r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) override;

  r2i::RuntimeError SetMemoryUsage (double memory_usage);

  r2i::RuntimeError Start () override;

  r2i::RuntimeError Stop () override;

  std::shared_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
      in_frame, r2i::RuntimeError &error) override;

  RuntimeError Predict (std::shared_ptr<r2i::IFrame> in_frame,
                        std::vector< std::shared_ptr<r2i::IPrediction> > &predictions) override;

  ~Engine ();

 private:
  enum State {
    STARTED,
    STOPPED
  };

  State state;
  int session_memory_usage_index;

  std::shared_ptr<TF_Session> session;
  std::shared_ptr<Model> model;
};

}
}
#endif //R2I_TENSORFLOW_ENGINE_H
