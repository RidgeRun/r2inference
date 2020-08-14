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

#include "loader.h"

#define MODULE_FACTORY_PREPROCESSING_SYMBOL "factory_make_preprocessing"
#define MODULE_FACTORY_POSTPROCESSING_SYMBOL "factory_make_postprocessing"

typedef
r2i::IPreprocessing *(*PreprocessFactoryFunc) (void);
typedef
r2i::IPostprocessing *(*PostprocessFactoryFunc) (void);

namespace r2i {

std::shared_ptr<r2i::IModel> Loader::Load (const std::string & /*in_path*/,
    r2i::RuntimeError &error) {
  error.Set(RuntimeError::NOT_IMPLEMENTED, "Loader::Load method not implemented");
  return nullptr;
}

std::shared_ptr<IPreprocessing> Loader::LoadPreprocessing(
  const std::string &in_path, RuntimeError &error) {
  PreprocessFactoryFunc factory;

  GModule *module = LoadModule(in_path.c_str(), error);
  if (RuntimeError::EOK != error.GetCode()) {
    return nullptr;
  }

  if (!g_module_symbol (module, MODULE_FACTORY_PREPROCESSING_SYMBOL,
                        (gpointer *) & factory)) {
    error.Set(RuntimeError::UNKNOWN_ERROR, g_module_error ());
    return nullptr;
  }

  IPreprocessing *preprocessing = (IPreprocessing *) factory();
  if (!preprocessing) {
    error.Set(RuntimeError::UNKNOWN_ERROR, "Error casting preprocessing object.");
    return nullptr;
  }

  return std::shared_ptr<IPreprocessing>(preprocessing);
}

std::shared_ptr<IPostprocessing> Loader::LoadPostprocessing(
  const std::string &in_path, RuntimeError &error) {
  PostprocessFactoryFunc factory;

  GModule *module = LoadModule(in_path.c_str(), error);
  if (RuntimeError::EOK != error.GetCode()) {
    return nullptr;
  }

  if (!g_module_symbol (module, MODULE_FACTORY_POSTPROCESSING_SYMBOL,
                        (gpointer *) & factory)) {
    error.Set(RuntimeError::UNKNOWN_ERROR, g_module_error ());
    return nullptr;
  }

  IPostprocessing *postprocessing = (IPostprocessing *) factory();
  if (!postprocessing) {
    error.Set(RuntimeError::UNKNOWN_ERROR, "Error casting postprocessing object.");
    return nullptr;
  }

  return std::shared_ptr<IPostprocessing>(postprocessing);
}

GModule *Loader::LoadModule(const gchar *in_path, RuntimeError &error) {
  GModule *module = nullptr;

  if (!in_path) {
    error.Set (RuntimeError::Code::NULL_PARAMETER,
               "Null input path for loading dynamic module.");
    return nullptr;
  }
  module = g_module_open (in_path, G_MODULE_BIND_LOCAL);

  if (!module) {
    error.Set (RuntimeError::Code::WRONG_API_USAGE,
               "Error loading dynamic module.");
    return nullptr;
  } else {
    return module;
  }
}

}
