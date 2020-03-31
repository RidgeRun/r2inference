# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_TFLITE

noinst_LTLIBRARIES = libtflite.la

# Define a custom area for our headers
tensorflowliteincludedir = @R2IINCLUDEDIR@/r2i/tensorflowlite

libtflite_la_SOURCES =           \
    engine.cc                    \
    frame.cc                     \
    loader.cc                    \
    model.cc                     \
    prediction.cc                \
    frameworkfactory.cc          \
    parameters.cc

libtflite_la_CPPFLAGS =              \
    $(RR_CPPFLAGS)                   \
    $(CODE_COVERAGE_CPPFLAGS)

libtflite_la_CFLAGS =                \
    $(RR_CFLAGS)                     \
    $(TFLITE_CFLAGS)         \
    $(CODE_COVERAGE_CFLAGS)

libtflite_la_CXXFLAGS =              \
    $(RR_CXXFLAGS)                   \
    $(TFLITE_CFLAGS)         \
    $(CODE_COVERAGE_CXXFLAGS)

libtflite_la_LIBADD =                \
    $(RR_LIBS)                       \
    $(TFLITE_LIBS)           \
    $(CODE_COVERAGE_LIBS)

libtflite_la_LDFLAGS =							 \
		-pthread -ldl -lrt

tensorflowliteinclude_HEADERS =      \
    model.h                          \
    loader.h                         \
    engine.h                         \
    prediction.h                     \
    frame.h                          \
    frameworkfactory.h               \
    parameters.h

endif # HAVE_TFLITE
