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

#ifndef R2I_TENSORFLOW_STATUSCODES_H
#define R2I_TENSORFLOW_STATUSCODES_H

#include <tensorflow/c/c_api.h>

#include <r2i/runtimeerror.h>

namespace r2i {
namespace tensorflow {

const std::string GetStringFromStatus (TF_Code status,
                                       r2i::RuntimeError &error);

} // namespace tensorflow
} // namespace r2k

#endif //R2I_TENSORFLOW_STATUSCODES_H