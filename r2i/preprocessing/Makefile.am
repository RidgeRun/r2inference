# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

lib_LTLIBRARIES =                             \
    libnormalize_inceptionv1.la               \
    libnormalize_inceptionv3.la

libnormalize_inceptionv1_la_SOURCES =         \
    normalize_inceptionv1.cc                  \
    normalize.cc

libnormalize_inceptionv3_la_SOURCES =         \
    normalize_inceptionv3.cc                  \
    normalize.cc

AM_CPPFLAGS =                                 \
    $(RR_CPPFLAGS)                            \
    $(CODE_COVERAGE_CPPFLAGS)                 \
    -I../../examples/r2i/common

AM_CFLAGS =                                   \
    $(RR_CFLAGS)                              \
    $(CODE_COVERAGE_CFLAGS)

AM_CXXFLAGS =                                 \
    $(RR_CXXFLAGS)                            \
    $(CODE_COVERAGE_CXXFLAGS)                 \
    $(GLIB_CFLAGS)                            \
    $(GMODULE_CFLAGS)

AM_LIBADD =                                   \
    $(RR_LIBS)                                \
    $(CODE_COVERAGE_LIBS)                     \
    $(GLIB_LIBS)                              \
    $(GMODULE_LIBS)

#Indicate libtool this is a module (dlopened library)
AM_CPPFLAGS_LDFLAGS =                         \
    -module -avoid-version -shared
