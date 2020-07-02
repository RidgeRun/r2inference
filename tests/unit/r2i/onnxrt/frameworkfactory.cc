/* Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
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
#include <r2i/onnxrt/frameworkfactory.h>

#include <memory>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/MemoryLeakDetectorMallocMacros.h>
#include <CppUTest/TestHarness.h>

#include <r2i/onnxrt/loader.h>

TEST_GROUP (OnnxFrameworkFactory) {
};

TEST (OnnxFrameworkFactory, ValidFactory) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IFrameworkFactory> factory;

  factory = r2i::IFrameworkFactory::MakeFactory(r2i::FrameworkCode::ONNXRT,
            error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Is this a valid ONNX Runtime factory? */
  auto ufactory =
    std::dynamic_pointer_cast<r2i::onnxrt::FrameworkFactory, r2i::IFrameworkFactory>
    (factory);
  CHECK (nullptr != ufactory);

  /* Test for loader */
  std::shared_ptr<r2i::ILoader> loader = factory->MakeLoader(error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  auto uloader =
    std::dynamic_pointer_cast<r2i::onnxrt::Loader, r2i::ILoader> (loader);
  CHECK (nullptr != uloader);

  /* Test for meta */
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
