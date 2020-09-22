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

#include "r2i/imageformat.h"

#include <string>
#include <unordered_map>

namespace r2i {

static std::unordered_map<int, std::pair<const std::string, int>>
format_descriptors ({
  {ImageFormat::Id::RGB, {"RGB", 3}},
  {ImageFormat::Id::BGR, {"BGR", 3}},
  {ImageFormat::Id::GRAY8, {"Grayscale", 1}},
  {ImageFormat::Id::UNKNOWN_FORMAT, {"Unknown format", 0}}
});

ImageFormat::ImageFormat ()
  : id(Id::UNKNOWN_FORMAT) {
}

ImageFormat::ImageFormat (Id id)
  : id(id) {
}

ImageFormat::Id ImageFormat::GetId () {
  return this->id;
}

std::pair<const std::string, int> Search (ImageFormat::Id id) {
  auto search = format_descriptors.find (id);
  if (format_descriptors.end () == search) {
    search = format_descriptors.find (ImageFormat::Id::UNKNOWN_FORMAT);
  }
  return search->second;
}

const std::string ImageFormat::GetDescription () {
  std::pair<const std::string, int> descriptor = Search(this->id);
  return descriptor.first;
}

int ImageFormat::GetNumPlanes () {
  std::pair<const std::string, int> descriptor = Search(this->id);
  return descriptor.second;
}
}
