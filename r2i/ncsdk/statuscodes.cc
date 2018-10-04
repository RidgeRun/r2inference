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
namespace ncsdk {

static std::unordered_map<int, const std::string> string_descriptions ({
  {NC_OK, "Everything OK"},
  {NC_BUSY, "Device is busy, retry later"},
  {NC_ERROR, "Error communicating with the device"},
  {NC_OUT_OF_MEMORY, "Out of memory"},
  {NC_DEVICE_NOT_FOUND, "No device at the given index or name"},
  {NC_INVALID_PARAMETERS, "At least one of the given parameters is wrong"},
  {NC_TIMEOUT, "Timeout in the communication with the device"},
  {NC_MVCMD_NOT_FOUND, "The file to boot Myriad was not found"},
  {NC_NOT_ALLOCATED, "The graph or device has been closed during the operation"},
  {NC_UNAUTHORIZED, "Unauthorized operation"},
  {NC_UNSUPPORTED_GRAPH_FILE, "The graph file version is not supported"},
  {NC_UNSUPPORTED_CONFIGURATION_FILE, "The configuration file version is not supported"},
  {NC_UNSUPPORTED_FEATURE, "Not supported by this FW version"},
  {NC_MYRIAD_ERROR, "An error has been reported by the device, use NC_DEVICE_DEBUG_INFO or NC_GRAPH_DEBUG_INFO"},
  {NC_INVALID_DATA_LENGTH, "invalid data length has been passed when get/set option"},
  {NC_INVALID_HANDLE, "handle to object that is invalid"}
});

const std::string GetStringFromStatus (ncStatus_t status,
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

} // namespace ncsdk
} // namespace r2k
