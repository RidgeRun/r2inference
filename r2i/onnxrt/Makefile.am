# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_ONNXRT

noinst_LTLIBRARIES = libonnxrt.la

# Define a custom area for our headers
onnxrtincludedir = @R2IINCLUDEDIR@/r2i/onnxrt

libonnxrt_la_SOURCES =           \
    engine.cc                    \
    frame.cc                     \
    frameworkfactory.cc          \
    loader.cc                    \
    model.cc                     \
    parameters.cc                \
    prediction.cc

libonnxrt_la_CPPFLAGS =          \
    $(RR_CPPFLAGS)               \
    $(CODE_COVERAGE_CPPFLAGS)

libonnxrt_la_CFLAGS =            \
    $(RR_CFLAGS)                 \
    $(ONNXRT_CFLAGS)             \
    $(CODE_COVERAGE_CFLAGS)

libonnxrt_la_CXXFLAGS =          \
    $(RR_CXXFLAGS)               \
    $(ONNXRT_CFLAGS)             \
    $(CODE_COVERAGE_CXXFLAGS)    \
    $(GLIB_CFLAGS)               \ 
    $(GMODULE_CFLAGS)

libonnxrt_la_LIBADD =            \
    $(RR_LIBS)                   \
    $(ONNXRT_LIBS)               \
    $(CODE_COVERAGE_LIBS)        \
    $(GLIB_LIBS)                 \ 
    $(GMODULE_LIBS)

onnxrtinclude_HEADERS =          \
    engine.h                     \
    frame.h                      \
    frameworkfactory.h           \
    loader.h                     \
    model.h                      \
    parameters.h                 \
    prediction.h

endif # HAVE_ONNXRT
