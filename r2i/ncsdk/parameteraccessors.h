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

#ifndef R2I_NCSDK_PARAMETER_ACCESSORS_H
#define R2I_NCSDK_PARAMETER_ACCESSORS_H

#include <r2i/runtimeerror.h>
#include <r2i/ncsdk/parameters.h>

namespace r2i {
namespace ncsdk {

RuntimeError SetParameterGlobal (Parameters *self, int param,
                                 void *target,
                                 unsigned int *target_size);
RuntimeError GetParameterGlobal (Parameters *self, int param,
                                 void *target,
                                 unsigned int *target_size);
RuntimeError GetParameterEngine (Parameters *self, int param,
                                 void *target,
                                 unsigned int *target_size);
RuntimeError SetParameterEngine (Parameters *self, int param,
                                 void *target,
                                 unsigned int *target_size);
RuntimeError GetParameterInputFifo (Parameters *self, int param,
                                    void *target,
                                    unsigned int *target_size);
RuntimeError SetParameterInputFifo (Parameters *self, int param,
                                    void *target,
                                    unsigned int *target_size);
RuntimeError GetParameterOutputFifo (Parameters *self, int param,
                                     void *target,
                                     unsigned int *target_size);
RuntimeError SetParameterOutputFifo (Parameters *self, int param,
                                     void *target,
                                     unsigned int *target_size);
RuntimeError SetParameterGraph (Parameters *self, int param,
                                void *target,
                                unsigned int *target_size);
RuntimeError GetParameterGraph (Parameters *self, int param,
                                void *target,
                                unsigned int *target_size);

} // namespace ncsdk
} // namespace r2k

#endif //R2I_NCSDK_PARAMETER_ACCESSORS_H
