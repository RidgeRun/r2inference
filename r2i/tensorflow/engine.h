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

#include <r2i/engine.h>

#include <memory>

#include <r2i/tensorflow/model.h>

namespace r2i {
namespace tensorflow {

class Engine : public r2i::Engine {
 public:
  Engine ();

  r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) override;

  r2i::RuntimeError SetMemoryUsage (double memory_usage);

  r2i::RuntimeError Start () override;

  r2i::RuntimeError Stop () override;

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
  virtual std::shared_ptr<r2i::IPrediction> Process (std::shared_ptr<r2i::IFrame>
    in_frame, r2i::RuntimeError &error) override;
};

}
}
#endif //R2I_TENSORFLOW_ENGINE_H
