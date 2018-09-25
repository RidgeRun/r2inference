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
#include <r2i/runtime_error.h>

#include <string>

/**
 * R2Inference Namespace
 */
namespace r2i {
  /**
   *  IParameter class class implements the interface to abstract IFrameworkFactory
   *  parameters. This Interface can be use to set and get parameters available
   *  in a framework factory. 
   */
  
  class IParameters {
    
 public:
    /**
     * \brief Method to query configure parameters available based on a IEngine and
     *  a IModel.
     * \param in_engine IEngine available for a IFrameworkFactory.
     * \param in_model IModel compatible with the IFrameworkFactory.
     * \param error a RuntimeError with a description of an error.
     * \return void .
     */    

    virtual void Configure (r2i::IEngine in_engine, r2i::IModel in_model,
                            RuntimeError &error ) {};

    /**
     * \brief Method to get the result of a string based parameter.
     * \param in_parameter Name of the parameter to get a value
     * \param value return value of the parameter to query
     * \param error a RuntimeError with a description of an error.
     * \return void.
     */        
    virtual void Get (std::string in_parameter, int &value , r2i::RuntimeError &error );


    /**
     * \brief Method to get the result of a int based parameter.
     * \param in_parameter Name of the parameter to get a value
     * \param value return value of the parameter to query
     * \param error a RuntimeError with a description of an error.     
     * \return void.
     */        
    virtual void Get (std::string in_parameter, std::string &value ,
                      r2i::RuntimeError &error );

    
    /**
     * \brief Method to set the value of a string based parameter.
     * \param in_parameter  Name of the parameter to set a value
     * \param in_value New value to set for in_parameter
     * \param error a RuntimeError with a description of an error.
     * \return void .
     */    
    virtual void Set (std::string in_parameter, std::string in_value, RuntimeError &error );

    /**
     * \brief Method to set the value of a int based parameter.
     * \param in_parameter Name of the parameter to set a value
     * \param in_value New value to set for in_parameter
     * \param error a RuntimeError with a description of an error.
     * \return void .
     */    
    virtual void Set (std::string in_parameter, int in_value, RuntimeError &error );

    
  };
  
}

#endif // R2I_IPARAMETERS_H
