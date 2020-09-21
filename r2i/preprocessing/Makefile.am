# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

lib_LTLIBRARIES = mean_std_preprocessing.la

mean_std_preprocessing_la_SOURCES =           \
    mean_std_preprocessing.cc

mean_std_preprocessing_la_CPPFLAGS =          \
    $(RR_CPPFLAGS)                            \
    $(CODE_COVERAGE_CPPFLAGS)                 \
    -I../../examples/r2i/common

mean_std_preprocessing_la_CFLAGS =            \
    $(RR_CFLAGS)                              \
    $(CODE_COVERAGE_CFLAGS)

mean_std_preprocessing_la_CXXFLAGS =          \
    $(RR_CXXFLAGS)                            \
    $(CODE_COVERAGE_CXXFLAGS)                 \
    $(GLIB_CFLAGS)                            \
    $(GMODULE_CFLAGS)

mean_std_preprocessing_la_LIBADD =            \
    $(RR_LIBS)                                \
    $(CODE_COVERAGE_LIBS)                     \
    $(GLIB_LIBS)                              \
    $(GMODULE_LIBS)

#Indicate libtool this is a module (dlopened library)
mean_std_preprocessing_la_LDFLAGS =           \
    -module -avoid-version -shared
