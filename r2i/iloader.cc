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

#include "iloader.h"

#define MODULE_FACTORY_SYMBOL "factory_make"

namespace r2i {

std::shared_ptr<IPreprocessing> ILoader::LoadPreprocessing(
  const std::string &in_path, RuntimeError &error) {
  GModule *module = nullptr;
  PreprocessFactoryFunc factory;

  error = LoadModule(in_path.c_str(), module);
  if (RuntimeError::EOK != error.GetCode()) {
    return nullptr;
  }

  if (!g_module_symbol (module, MODULE_FACTORY_SYMBOL,
                        (gpointer *) & factory)) {
    error.Set(RuntimeError::UNKNOWN_ERROR, "Error loading preprocessing symbols.");
    return nullptr;
  }

  IPreprocessing *preprocessing = (IPreprocessing *) factory();
  if (!preprocessing) {
    error.Set(RuntimeError::UNKNOWN_ERROR, "Error casting preprocessing object.");
    return nullptr;
  }

  g_free (module);

  return std::shared_ptr<IPreprocessing>(preprocessing);
}

std::shared_ptr<IPostprocessing> ILoader::LoadPostprocessing(
  const std::string &in_path, RuntimeError &error) {
  GModule *module = nullptr;
  PostprocessFactoryFunc factory;

  error = LoadModule(in_path.c_str(), module);
  if (RuntimeError::EOK != error.GetCode()) {
    return nullptr;
  }

  if (!g_module_symbol (module, MODULE_FACTORY_SYMBOL,
                        (gpointer *) & factory)) {
    error.Set(RuntimeError::UNKNOWN_ERROR, "Error loading postprocessing symbols.");
    return nullptr;
  }

  IPostprocessing *postprocessing = (IPostprocessing *) factory();
  if (!postprocessing) {
    error.Set(RuntimeError::UNKNOWN_ERROR, "Error casting postprocessing object.");
    return nullptr;
  }

  g_free (module);

  return std::shared_ptr<IPostprocessing>(postprocessing);
}

RuntimeError ILoader::LoadModule(const gchar *in_path, GModule *module) {
  RuntimeError error;
  gchar *filename = nullptr;

  if (!in_path) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Null input path for loading dynamic module.");
    return error;
  }

  filename = g_filename_from_uri (in_path, nullptr, nullptr);
  module = g_module_open (filename, (GModuleFlags) G_MODULE_BIND_LOCAL);
  g_free (filename);

  if (!module) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Error loading dynamic module.");
    return error;
  } else {
    return error;
  }
}
}