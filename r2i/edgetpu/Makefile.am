# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_EDGETPU

noinst_LTLIBRARIES = libedgetpu.la

# Define a custom area for our headers
tensorflowliteedgetpuincludedir = @R2IINCLUDEDIR@/r2i/edgetpu

libedgetpu_la_SOURCES =                     \
    engine.cc                               \
    frameworkfactory.cc 

libedgetpu_la_CPPFLAGS =                    \
    $(RR_CPPFLAGS)                          \
    $(CODE_COVERAGE_CPPFLAGS)

libedgetpu_la_CFLAGS =                      \
    $(RR_CFLAGS)                            \
    $(TFLITE_CFLAGS)                        \
    $(EDGETPU_CFLAGS)                       \
    $(CODE_COVERAGE_CFLAGS)

libedgetpu_la_CXXFLAGS =                    \
    $(RR_CXXFLAGS)                          \
    $(TFLITE_CFLAGS)                        \
    $(EDGETPU_CFLAGS)                       \
    $(CODE_COVERAGE_CXXFLAGS)               \
    $(GLIB_CFLAGS)                          \ 
    $(GMODULE_CFLAGS)

libedgetpu_la_LIBADD =                      \
    $(RR_LIBS)                              \
    $(TFLITE_LIBS)                          \
    $(EDGETPU_LIBS)                         \
    $(CODE_COVERAGE_LIBS)                   \
    $(GLIB_LIBS)                            \ 
    $(GMODULE_LIBS)

tensorflowliteedgetpuinclude_HEADERS =      \
    engine.h                                \
    frameworkfactory.h

endif # HAVE_EDGETPU
