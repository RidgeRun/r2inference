/* Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#include "r2i/datatype.h"

#include <string>
#include <unordered_map>

namespace r2i {

static std::unordered_map<int, std::pair<const std::string, int>>
format_descriptors ({
  {DataType::Id::INT32, {"32 bit integer", 3}},
  {DataType::Id::FLOAT, {"32 bit float", 3}},
  {DataType::Id::HALF, {"16 bit float", 2}},
  {DataType::Id::BOOL, {"8 bit boolean", 1}},
  {DataType::Id::INT8, {"8 bit integer", 1}},
  {DataType::Id::UNKNOWN_DATATYPE, {"Unknown format", 0}}
});

DataType::DataType ()
  : id(Id::UNKNOWN_DATATYPE) {
}

DataType::DataType (Id id)
  : id(id) {
}

DataType::Id DataType::GetId () {
  return this->id;
}

std::pair<const std::string, int> search (DataType::Id id) {
  auto search = format_descriptors.find (id);
  if (format_descriptors.end () == search) {
    search = format_descriptors.find (DataType::Id::UNKNOWN_DATATYPE);
  }
  return search->second;
}

const std::string DataType::GetDescription () {
  std::pair<const std::string, int> descriptor = search(this->id);
  return descriptor.first;
}

int DataType::GetBytesPerPixel () {
  std::pair<const std::string, int> descriptor = search(this->id);
  return descriptor.second;
}
}
