# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_CORAL

noinst_LTLIBRARIES = libcoral.la

# Define a custom area for our headers
tensorflowlitecoralincludedir = @R2IINCLUDEDIR@/r2i/coral

libcoral_la_SOURCES =                       \
    engine.cc                               \
    frameworkfactory.cc 

libcoral_la_CPPFLAGS =                      \
    $(RR_CPPFLAGS)                          \
    $(CODE_COVERAGE_CPPFLAGS)

libcoral_la_CFLAGS =                        \
    $(RR_CFLAGS)                            \
    $(TFLITE_CFLAGS)                        \
    $(CORAL_CFLAGS)                         \
    $(CODE_COVERAGE_CFLAGS)

libcoral_la_CXXFLAGS =                      \
    $(RR_CXXFLAGS)                          \
    $(TFLITE_CFLAGS)                        \
    $(CORAL_CFLAGS)                         \
    $(CODE_COVERAGE_CXXFLAGS)               \
    $(GLIB_CFLAGS)                          \ 
    $(GMODULE_CFLAGS)

libcoral_la_LIBADD =                        \
    $(RR_LIBS)                              \
    $(TFLITE_LIBS)                          \
    $(CORAL_LIBS)                           \
    $(CODE_COVERAGE_LIBS)                   \
    $(GLIB_LIBS)                            \ 
    $(GMODULE_LIBS)

tensorflowlitecoralinclude_HEADERS =        \
    engine.h                                \
    frameworkfactory.h

endif # HAVE_CORAL
