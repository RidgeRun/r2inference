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

#include <vector>

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
   * \param output_index corresponding to the model output.
   * \param index Index for a value on the prediction matrix.
   * \param error [out] RuntimeError with a description of an error.
   * \return a double that indicates the prediction at the provided index .
   */
  virtual double At (unsigned int output_index, unsigned int index,
                     r2i::RuntimeError &error) = 0;

  /**
   * \brief Set predictions result
   * \param data obtained from the model inference.
   * \param size of the input data.
   * \return Error code resulting from the set process.
   */
  virtual r2i::RuntimeError AddResult (float *data, unsigned int size) = 0;

  /**
   * \brief Set a prediction result in some specific index.
   * \param output_index Index of the result data.
   * \param data predicted.
   * \param size of the data array.
   * \return Error code resulting from the insert process.
   */
  virtual r2i::RuntimeError InsertResult (unsigned int output_index, float *data,
                                          unsigned int size) = 0;

  /**
   * \brief Gets the underlying vector to the result data. The pointer
   * will be valid as long as the prediction is valid. The actual type
   * of the underlying data will depend on the backend being used.
   * \param output_index corresponding to the model output.
   * \param error [out] RuntimeError with a description of an error.
   * \return The underlying result pointer.
   */
  virtual void *GetResultData (unsigned int output_index,
                               RuntimeError &error) = 0;

  /**
   * \brief Gets the size (in bytes) of the underlying data. Note that
   * this size wont necessarily match with the number of elements. To
   * get the number of elements, an explicit division must be made
   * such as:
   *   int num_of_elements = pred->GetResultSize () / sizeof (element);
   * where sizeof (element) expands to the size of a single element.
   * \param output_index corresponding to the model output.
   * \param error [out] RuntimeError with a description of an error.
   * \return The underlying result size in bytes
   */
  virtual unsigned int GetResultSize (unsigned int output_index,
                                      RuntimeError &error) = 0;

  /**
   * \brief Gets the number of output predictions stored.
   * \return Integer value with the number of output predictions.
   */
  virtual unsigned int GetOutputCount() = 0;

  /**
   * \brief Default destructor
   */
  virtual ~IPrediction () {};
};

}

#endif // R2I_IPREDICTION_H
