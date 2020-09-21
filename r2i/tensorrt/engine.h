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

#ifndef R2I_TENSORRT_ENGINE_H
#define R2I_TENSORRT_ENGINE_H

#include <memory>

#include <r2i/iengine.h>
#include <r2i/tensorrt/model.h>

namespace r2i {
namespace tensorrt {

class Engine : public IEngine {
 public:
  Engine ();

  r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) override;

  r2i::RuntimeError Start () override;

  r2i::RuntimeError Stop () override;

  std::shared_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
      in_frame, r2i::RuntimeError &error) override;

  RuntimeError Predict (std::shared_ptr<r2i::IFrame> in_frame,
                        std::vector< std::shared_ptr<r2i::IPrediction> > &predictions) override;

  r2i::RuntimeError SetBatchSize (const int batchsize);

  const int GetBatchSize ();

  ~Engine ();

 private:
  std::shared_ptr<Model> model;
  int batch_size;
};

}
}
#endif //R2I_TENSORRT_ENGINE_H
