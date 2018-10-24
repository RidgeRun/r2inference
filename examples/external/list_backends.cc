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

#include <iostream>
#include <r2i/r2i.h>

void PrintFramework (r2i::FrameworkMeta &meta) {
  std::cout << "Name        : " << meta.name << std::endl;
  std::cout << "Description : " << meta.description << std::endl;
  std::cout << "Version     : " << meta.version << std::endl;
  std::cout << "---" << std::endl;
}

int main (int argc, char *argv[]) {
  r2i::RuntimeError error;

  std::cout << "Backends supported by your system:" << std::endl;
  std::cout << "==================================" << std::endl;

  for (auto &meta : r2i::IFrameworkFactory::List (error)) {
    PrintFramework (meta);
  }

  return 0;
}

