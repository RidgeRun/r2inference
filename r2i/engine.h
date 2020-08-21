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

#ifndef R2I_ENGINE_H
#define R2I_ENGINE_H

#include <r2i/iengine.h>

/**
 * R2Inference Namespace
 */
namespace r2i {

class Engine : public IEngine {
 public:

  virtual r2i::RuntimeError SetModel (std::shared_ptr<r2i::IModel> in_model) override;
  virtual r2i::RuntimeError Start () override;
  virtual r2i::RuntimeError Stop () override;
  virtual std::shared_ptr<r2i::IPrediction> Predict (std::shared_ptr<r2i::IFrame>
      in_frame, r2i::RuntimeError &error) override;
  virtual RuntimeError SetPreprocessing (std::shared_ptr<IPreprocessing> preprocessing) override;
  virtual std::shared_ptr<IPreprocessing> GetPreprocessing () override;
  virtual RuntimeError SetPostprocessing (std::shared_ptr<IPostprocessing> postprocessing) override;
  virtual std::shared_ptr<IPostprocessing> GetPostprocessing () override;

 private:
  std::shared_ptr<IPreprocessing> preprocessing;
  std::shared_ptr<IPostprocessing> postprocessing;
};

}

#endif // R2I_ENGINE_H
