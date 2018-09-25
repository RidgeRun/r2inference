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

#ifndef R2I_IFRAMEWORKFACTORY_H
#define R2I_IFRAMEWORKFACTORY_H

#include <r2i/iloader.h>
#include <r2i/imodel.h>
#include <r2i/iengine.h>
#include <r2i/iparameters.h>
#include <r2i/frameworkmeta.h>

#include <vector>
#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
  /**
   *  IFrameworkFactory class implements the interface to abstract factory to
   *  create framework related objects
   */
  
  class IFrameworkFactory {
    
 public:
    /**
     * \brief Method to autodetect a Framework factory available on the system 
     * \param error a RuntimeError with a description of an error.
     * \return a valid Frameworkfactory.
     */
    static r2i::IFrameworkFactory Autodetect(RuntimeError &error){};

    /**
     * \brief Method to check all the available Frameworks on system .
     * \param error a RuntimeError with a description of an error.
     * \return List of supported Frameworks.
     */
    static std::vector<r2i::IFrameworkFactory> List(RuntimeError &error) {};
    
    /**
     * \brief Method to create a ILoader based on a particular Framework
     * \param error a RuntimeError with a description of an error.
     * \return a valid Loader for the Framework.
     */
    virtual r2i::ILoader MakeLoader (RuntimeError &error) {};
    /**
     * \brief Method to get a valid IEngine based on a particular Framework
     * \param error a RuntimeError with a description of an error.
     * \return a valid IEngine for the framework.
     */
    
    virtual r2i::IEngine MakeEngine (RuntimeError &error) {};
    /**
     * \brief Method to get the valid IParameters on a particular Framework
     * \param error a RuntimeError with a description of an error.
     * \return a valid Parameters of a framework.
     */
    
    virtual r2i::IParameters MakeParameters (RuntimeError &error) {};
    /**
     * \brief Method to get the Metadata of particular Framework
     * \param error a RuntimeError with a description of an error.
     * \return valid Metadata of a Framework.
     */
    
    virtual r2i::FrameworkMeta GetDescription(RuntimeError &error) {};
  };
  
}

#endif // R2I_IFRAMEWORKFACTORY_H
