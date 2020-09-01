# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_ONNXRT_OPENVINO

noinst_LTLIBRARIES = libonnxrt_openvino.la

# Define a custom area for our headers
onnxrt_openvinoincludedir = @R2IINCLUDEDIR@/r2i/onnxrt_openvino

libonnxrt_openvino_la_SOURCES =             \
    engine.cc                               \
    frameworkfactory.cc                     \
    parameters.cc                           \
    parameteraccessors.cc

libonnxrt_openvino_la_CPPFLAGS =            \
    $(RR_CPPFLAGS)                          \
    $(CODE_COVERAGE_CPPFLAGS)

libonnxrt_openvino_la_CFLAGS =              \
    $(RR_CFLAGS)                            \
    $(ONNXRT_CFLAGS)                        \
    $(ONNXRT_OPENVINO_CFLAGS)               \
    $(CODE_COVERAGE_CFLAGS)

libonnxrt_openvino_la_CXXFLAGS =            \
    $(RR_CXXFLAGS)                          \
    $(ONNXRT_CFLAGS)                        \
    $(ONNXRT_OPENVINO_CFLAGS)               \
    $(CODE_COVERAGE_CXXFLAGS)               \
    $(GLIB_CFLAGS)                          \ 
    $(GMODULE_CFLAGS)

libonnxrt_openvino_la_LIBADD =              \
    $(RR_LIBS)                              \
    $(ONNXRT_LIBS)                          \
    $(ONNXRT_OPENVINO_LIBS)                 \
    $(CODE_COVERAGE_LIBS)                   \
    $(GLIB_LIBS)                            \ 
    $(GMODULE_LIBS)

onnxrt_openvinoinclude_HEADERS =            \
    engine.h                                \
    frameworkfactory.h                      \
    parameters.h                            \
    parameteraccessors.h

endif # HAVE_ONNXRT_OPENVINO
