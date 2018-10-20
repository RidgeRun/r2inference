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

#include <memory>

#include <r2i/iframeworkfactory.h>
#include <r2i/ncsdk/frameworkfactory.h>
#include <r2i/ncsdk/loader.h>
#include <r2i/ncsdk/engine.h>
#include <r2i/ncsdk/parameters.h>

#include <CppUTest/CommandLineTestRunner.h>
#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>

TEST_GROUP (NcsdkFrameworkFactory) {
};

TEST (NcsdkFrameworkFactory, ValidFactory) {
  r2i::RuntimeError error;
  std::shared_ptr<r2i::IFrameworkFactory> factory;

  factory = r2i::IFrameworkFactory::MakeFactory(
              r2i::FrameworkCode::NCSDK, error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());

  /* Is this a valid NCSDK factory? */
  auto ufactory =
    std::dynamic_pointer_cast<r2i::ncsdk::FrameworkFactory, r2i::IFrameworkFactory>
    (factory);
  CHECK (nullptr != ufactory);

  /* Test for loader */
  std::shared_ptr<r2i::ILoader> loader = factory->MakeLoader(error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  auto uloader = std::dynamic_pointer_cast<r2i::ncsdk::Loader, r2i::ILoader>
                 (loader);
  CHECK (nullptr != uloader);

  /* Test for engine */
  std::shared_ptr<r2i::IEngine> engine = factory->MakeEngine(error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  auto uengine = std::dynamic_pointer_cast<r2i::ncsdk::Engine, r2i::IEngine>
                 (engine);
  CHECK (nullptr != uengine);

  /* Test for parameters */
  std::shared_ptr<r2i::IParameters> parameters = factory->MakeParameters(error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
  auto uparameters =
    std::dynamic_pointer_cast<r2i::ncsdk::Parameters, r2i::IParameters>(parameters);
  CHECK (nullptr != uparameters);

  /* Test for meta */
  auto description = factory->GetDescription (error);
  LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

int main (int ac, char **av) {
  return CommandLineTestRunner::RunAllTests (ac, av);
}
