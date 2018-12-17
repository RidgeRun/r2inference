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

#include "statuscodes.h"

#include <string>
#include <unordered_map>

namespace r2i {
namespace tensorflow {

static std::unordered_map<int, const std::string> string_descriptions ({
  {TF_OK, "Everything OK"},
  {TF_CANCELLED, "Operation cancelled"},
  {TF_UNKNOWN, "Unknown error"},
  {TF_INVALID_ARGUMENT, "Invalid argument"},
  {TF_DEADLINE_EXCEEDED, "Deadline exceeded"},
  {TF_NOT_FOUND, "Resource not found"},
  {TF_ALREADY_EXISTS, "Resource already exists"},
  {TF_PERMISSION_DENIED, "Permission denied"},
  {TF_UNAUTHENTICATED, "Unauthenticated"},
  {TF_RESOURCE_EXHAUSTED, "Resource exhausted"},
  {TF_FAILED_PRECONDITION, "Precondition failed"},
  {TF_ABORTED, "Operation aborted"},
  {TF_OUT_OF_RANGE, "Out of range"},
  {TF_UNIMPLEMENTED, "Operation unimplemented"},
  {TF_INTERNAL, "Internal error"},
  {TF_UNAVAILABLE, "Unavailable"},
  {TF_DATA_LOSS, "Data loss"},
});

const std::string GetStringFromStatus (TF_Code status,
                                       r2i::RuntimeError &error) {
  std::string ret;

  error.Clean ();

  auto search = string_descriptions.find (status);

  if (string_descriptions.end () == search) {
    ret = "Unable to find enum value, outdated list";
    error.Set (RuntimeError::Code::UNKNOWN_ERROR, ret);
  } else {
    ret = search->second;
  }

  return ret;
}

} // namespace tensorflow
} // namespace r2k
