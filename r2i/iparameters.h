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

#ifndef R2I_IPARAMETERS_H
#define R2I_IPARAMETERS_H

#include <r2i/iengine.h>
#include <r2i/imodel.h>
#include <r2i/runtimeerror.h>

#include <memory>
#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
/**
 *  Implements the interface to abstract IFrameworkFactory parameters. This
 *  interface can be use to set and get parameters available in a framework
 *  factory.
 */

class IParameters {

 public:
  /**
   * \brief Sets an IEngine and IModel as targets for the parameters accessors.
   * \param in_engine IEngine available for a IFrameworkFactory.
   * \param in_model IModel compatible with the IFrameworkFactory.
   * \param error [out] RuntimeError with a description of an error.
   */
  virtual void Configure (std::shared_ptr<r2i::IEngine> in_engine,
                          std::shared_ptr<r2i::IModel> in_model,
                          RuntimeError &error ) = 0;

  /**
   * \brief Gets the IEngine currently configured, if any.
   * \param error [out] RuntimeError with a description of an error.
   * \return A shared pointer to the engine currently configured.
   */
  virtual std::shared_ptr<r2i::IEngine> GetEngine (RuntimeError &error ) = 0;

  /**
   * \brief Gets the IModel currently configured, if any.
   * \param error [out] RuntimeError with a description of an error.
   * \return A shared pointer to the model currently configured.
   */
  virtual std::shared_ptr<r2i::IModel> GetModel (RuntimeError &error ) = 0;

  /**
   * \brief Queries an integer parameter.
   * \param in_parameter Name of the parameter to get a value
   * \param value [out] Return value of the parameter to query
   * \param error [out] RuntimeError with a description of an error.
   */
  virtual void Get (const std::string in_parameter, int &value,
                    r2i::RuntimeError &error ) = 0;

  /**
   * \brief Queries a string parameter.
   * \param in_parameter Name of the parameter to get a value
   * \param value [out] Return value of the parameter to query
   * \param error [out] RuntimeError with a description of an error.
   */
  virtual void Get (const std::string in_parameter, const std::string &value,
                    r2i::RuntimeError &error ) = 0;

  /**
   * \brief Sets a string parameter.
   * \param in_parameter  Name of the parameter to set a value
   * \param in_value New value to set for in_parameter
   * \param error [out] RuntimeError with a description of an error.
   */
  virtual void Set (const std::string in_parameter, const std::string &in_value,
                    RuntimeError &error ) = 0;

  /**
   * \brief Sets an integer parameter.
   * \param in_parameter Name of the parameter to set a value
   * \param in_value New value to set for in_parameter
   * \param error [out] RuntimeError with a description of an error.
   */
  virtual void Set (const std::string in_parameter, int in_value,
                    RuntimeError &error ) = 0;
};

}

#endif // R2I_IPARAMETERS_H
