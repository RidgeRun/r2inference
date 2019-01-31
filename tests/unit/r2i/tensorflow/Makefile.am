# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

AM_DEFAULT_SOURCE_EXT = .cc

if ENABLE_TESTS
if HAVE_TENSORFLOW

TESTS =            \
        loader     \
        parameters \
        engine     \
        model      \
        prediction \
        frame

check_PROGRAMS=$(TESTS)

endif # HAVE_TENSORFLOW
endif # ENABLE_TESTS

if ENABLE_CODE_COVERAGE
SINGLE_PROCESS=--single-process yes
else
SINGLE_PROCESS=
endif

LOG_DRIVER=$(top_builddir)/common/rr-test-driver.py $(SINGLE_PROCESS)

AM_CXXFLAGS =                       \
        $(RR_CXXFLAGS)              \
        $(TESTS_CFLAGS)             \
        $(TENSORFLOW_CFLAGS)        \
        $(CODE_COVERAGE_CXXFLAGS)

AM_CFLAGS =                         \
        $(RR_CFLAGS)                \
        $(TESTS_CFLAGS)             \
        $(TENSORFLOW_CFLAGS)        \
        $(CODE_COVERAGE_CFLAGS)

AM_CPPFLAGS =                       \
        $(RR_CPPFLAGS)              \
        $(CODE_COVERAGE_CPPFLAGS)

AM_LDFLAGS =                        \
        $(RR_LIBS)                  \
        $(TENSORFLOW_LIBS)          \
        $(TESTS_LIBS)               \
        $(CODE_COVERAGE_LIBS)

COMMON_SOURCES =                                   \
        @top_srcdir@/r2i/runtimeerror.cc           \
        @top_srcdir@/r2i/imageformat.cc

loader_SOURCES =                                   \
        loader.cc                                  \
        @top_srcdir@/r2i/tensorflow/loader.cc      \
        $(COMMON_SOURCES)

parameters_SOURCES =                               \
        parameters.cc                              \
        @top_srcdir@/r2i/tensorflow/parameters.cc  \
        $(COMMON_SOURCES)

engine_SOURCES =                                   \
        engine.cc                                  \
        @top_srcdir@/r2i/tensorflow/engine.cc      \
        $(COMMON_SOURCES)

model_SOURCES =                                   \
        model.cc                                  \
        @top_srcdir@/r2i/tensorflow/model.cc      \
        $(COMMON_SOURCES)

prediction_SOURCES =                              \
        prediction.cc                             \
        @top_srcdir@/r2i/tensorflow/prediction.cc \
        $(COMMON_SOURCES)

frame_SOURCES =                              \
        frame.cc                             \
        @top_srcdir@/r2i/tensorflow/frame.cc \
        $(COMMON_SOURCES)

CLEANFILES = cpputest_*.xml
