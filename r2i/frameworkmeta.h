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

#ifndef R2I_FRAMEWORK_META_H
#define R2I_FRAMEWORK_META_H

#include <string>

#include <r2i/frameworks.h>

/**
 * R2Inference Namespace
 */
namespace r2i {

/**
 * Implements the placeholder for framework information.
 */
struct FrameworkMeta {
  /**
   * The numerical code to create a framework factory
   */
  const FrameworkCode code;

  /**
   * A string to identify the name of the framework
   */
  const std::string name;

  /**
   * A short string to identify the framework
   */
  const std::string label;

  /**
  * A string with a description for the framework
  */
  const std::string description;

  /**
  * A string with the version descriptor for the framework
  */
  const std::string version;
};

}

#endif // R2I_FRAMEWORK_META_H
