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

#include <r2i/iframeworkfactory.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

TEST_GROUP (IFrameworkFactory) {
};

TEST (IFrameworkFactory, InvalidFramework) {
  r2i::RuntimeError error;
  std::unique_ptr<r2i::IFrameworkFactory> fw;

  fw = r2i::IFrameworkFactory::MakeFactory (
         r2i::IFrameworkFactory::FrameworkCode::MAX_FRAMEWORK, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::UNSUPPORTED_FRAMEWORK, error.GetCode ());
  POINTERS_EQUAL (nullptr, fw.get());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
