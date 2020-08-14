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

#ifndef R2I_TENSORFLOW_LOADER_H
#define R2I_TENSORFLOW_LOADER_H

#include <r2i/imodel.h>
#include <r2i/loader.h>
#include <r2i/tensorflow/model.h>

namespace r2i {
namespace tensorflow {

class Loader : public r2i::Loader {
 public:
  virtual std::shared_ptr<r2i::IModel> Load (const std::string &in_path,
      r2i::RuntimeError &error) override;
};

}
}

#endif //R2I_TENSORFLOW_LOADER_H
