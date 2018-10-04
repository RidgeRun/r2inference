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

#include <mvnc.h>
#include <r2i/r2i.h>
#include <r2i/ncsdk/engine.h>

#include <CppUTest/MemoryLeakDetectorNewMacros.h>
#include <CppUTest/TestHarness.h>


class MockModel : public r2i::IModel {
};

/* Stubs for MVNC */
bool engineerror = false;

TEST_GROUP (NcsdkEngine) {
    r2i::ncsdk::Engine engine;
    std::shared_ptr<r2i::IModel> model;

    void setup () {
        engineerror = false;
        model = std::make_shared<MockModel> ();
    }

    void teardown () {
    }
};

TEST (NcsdkEngine, SetModel) {
    r2i::RuntimeError error;

    engine.SetModel (model,error);
    LONGS_EQUAL (r2i::RuntimeError::Code::EOK, error.GetCode ());
}

TEST (NcsdkEngine, SetModelNull) {
    r2i::RuntimeError error;

    engine.SetModel (nullptr,error);
    LONGS_EQUAL (r2i::RuntimeError::Code::NULL_PARAMETER, error.GetCode ());
}
