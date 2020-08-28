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

#ifndef R2I_LOADER_H
#define R2I_LOADER_H

#include <r2i/iloader.h>

#include <glib.h>
#include <gmodule.h>

/**
 * R2Inference Namespace
 */
namespace r2i {

class Loader : public ILoader {
 public:
  virtual std::shared_ptr<r2i::IModel> Load (const std::string &in_path,
      r2i::RuntimeError &error) override;

  virtual std::shared_ptr<IPreprocessing> LoadPreprocessing (
    const std::string &in_path,
    RuntimeError &error) override;

  virtual std::shared_ptr<IPostprocessing> LoadPostprocessing (
    const std::string &in_path,
    RuntimeError &error) override;

  ~Loader () {};

 private:
  /**
   * \brief Load dynamic libraries modules.
   * \param in_path A string with the path to the dynamic library.
   * \param error [out] RuntimeError with a description of an error.
   * \return Loaded module or Null if the loading failed.
   */
  GModule *LoadModule(const gchar *in_path, RuntimeError &error);
};

}

#endif // R2I_ILOADER_H
