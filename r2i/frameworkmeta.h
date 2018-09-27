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

#ifndef R2I_FRAMEWORKMETA_H
#define R2I_FRAMEWORKMETA_H

#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i
{
/**
 * Implements the placeholder for framework information.
 */
class FrameworkMeta
{

public:
  /**
   * A string to identify the name of the framework
   */
  const std::string name;

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

#endif // R2I_FRAMEWORKMETA_H
