# Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_TENSORRT

noinst_LTLIBRARIES = libtensorrt.la

# Define a custom area for our headers
tensorrtincludedir = @R2IINCLUDEDIR@/r2i/tensorrt

libtensorrt_la_SOURCES =         \
    engine.cc                    \
    frame.cc                     \
    frameworkfactory.cc          \
    loader.cc                    \
    model.cc                     \
    parameters.cc                \
    prediction.cc

libtensorrt_la_CPPFLAGS =        \
    $(RR_CPPFLAGS)               \
    $(CODE_COVERAGE_CPPFLAGS)    \
    -I$(CUDA_INCDIR)

libtensorrt_la_CFLAGS =          \
    $(RR_CFLAGS)                 \
    $(CODE_COVERAGE_CFLAGS)

libtensorrt_la_CXXFLAGS =        \
    $(RR_CXXFLAGS)               \
    $(TENSORRT_CFLAGS)           \
    $(CODE_COVERAGE_CXXFLAGS)    \
    $(CUDA_LIBS)

libtensorrt_la_LIBADD =          \
    $(RR_LIBS)                   \
    $(TENSORRT_LIBS)             \
    $(CODE_COVERAGE_LIBS)        \
    $(CUDA_LIBS)

tensorrtinclude_HEADERS =        \
    engine.h                     \
    frame.h                      \
    frameworkfactory.h           \
    loader.h                     \
    model.h                      \
    parameters.h                 \
    prediction.h

endif # HAVE_TENSORRT
