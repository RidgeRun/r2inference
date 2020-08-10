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

#ifndef R2I_IPREPROCESSING_H
#define R2I_IPREPROCESSING_H

#include <r2i/iframe.h>
#include <r2i/imageformat.h>
#include <r2i/runtimeerror.h>

#include <vector>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the interface to validate an IPreprocessing implementation
 */
class IPreprocessing {

 public:
  /*
   */
  virtual RuntimeError apply(IFrame &data) = 0;

  /*
   */
  virtual std::vector<ImageFormat> getAvailableFormats() = 0;

  /*
   */
  virtual std::vector<std::vector<int>> getAvailableDataSizes() = 0;

  /**
   * \brief Default destructor
   */
  virtual ~IPreprocessing () {};
};


}

#endif // R2I_IPREPROCESSING_H
