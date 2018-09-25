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

/**
 * R2Inference Namespace
 */
namespace r2i {
  /**
   * FrameworkMeta class implements the placeholder for framework
   * informtation
   */
  class FrameworkMeta {

  public:
    std::string name_;
    std::string description_;
    std::string version_;
  };
  
}

#endif // R2I_FRAMEWORKMETA_H
