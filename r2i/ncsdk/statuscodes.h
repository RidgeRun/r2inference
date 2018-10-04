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

#ifndef R2I_NCSDK_STATUSCODES_H
#define R2I_NCSDK_STATUSCODES_H

#include <mvnc.h>
#include <string>

#include <r2i/runtimeerror.h>

namespace r2i {
namespace ncsdk {

const std::string GetStringFromStatus (ncStatus_t status,
                                       r2i::RuntimeError &error);

} // namespace ncsdk
} // namespace r2k

#endif //R2I_NCSDK_STATUSCODES_H
