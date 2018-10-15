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

#ifndef R2I_NCSDK_MODEL_H
#define R2I_NCSDK_MODEL_H

#include <r2i/imodel.h>
#include <r2i/runtimeerror.h>

#include <string>
#include <mvnc.h>
#include <memory>

namespace r2i {
namespace ncsdk {

class Model : public IModel {
 public:
  RuntimeError Start (const std::string &name) override;

  RuntimeError Stop ();

  ncGraphHandle_t *GetHandler ();

  std::shared_ptr<void> GetData ();
  void SetData (std::shared_ptr<void> graph_data);
  unsigned int GetDataSize ();
  void SetDataSize (unsigned int graph_size);

 private:
  ncGraphHandle_t *graph_handler;
  std::shared_ptr<void> graph_data;
  unsigned int graph_size;
};

}
}

#endif //R2I_NCSDK_MODEL_H
