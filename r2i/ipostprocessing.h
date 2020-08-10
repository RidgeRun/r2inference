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

#ifndef R2I_IPOSTPROCESSING_H
#define R2I_IPOSTPROCESSING_H

#include <r2i/runtimeerror.h>
#include <r2i/iprediction.h>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the interface to validate an IPostprocessing implementation
 */
class IPostprocessing {

 public:
  /*
   */
  virtual RuntimeError apply(IPrediction &prediction) = 0;

  /**
   * \brief Default destructor
   */
  virtual ~IPostprocessing () {};

};

}

#endif // R2I_IPOSTPROCESSING_H
