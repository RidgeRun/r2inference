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

#ifndef R2I_IPREDICTION_H
#define R2I_IPREDICTION_H

#include <r2i/runtimeerror.h>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 * Implements the interface to abstract the result of evaluation
 * an IFrame on an IModel.
 */

class IPrediction {
 public:
  /**
   * \brief Gets the prediction for a particular index.
   * \param index Index for a value on the prediction matrix.
   * \param error [out] RuntimeError with a description of an error.
   * \return a double that indicates the prediction at the provided index .
   */
  virtual float At (unsigned int index,  r2i::RuntimeError &error) = 0;

  /**
   * \brief Gets the underlying vector to the result data. The pointer
   * will be valid as long as the prediction is valid.
   * \return The underlying result pointer.
   */
  virtual float *GetResultData () = 0;

  /**
   * \brief Gets the size (in bytes) of the underlying data. Note that
   * this size wont necessarily match with the number of elements. To
   * get the number of elements, an explicit division must be made
   * such as:
   *   int num_of_elements = pred->GetResultSize () / sizeof (element);
   * where sizeof (element) expands to the size of a single element.
   *
   * \return The underlying result size in bytes
   */
  virtual unsigned int GetResultSize () = 0;

  /**
   * \brief Default destructor
   */
  virtual ~IPrediction () {};
};

}

#endif // R2I_IPREDICTION_H
