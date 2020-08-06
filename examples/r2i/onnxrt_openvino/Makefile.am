# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

AM_DEFAULT_SOURCE_EXT = .cc

if ENABLE_EXAMPLES

noinst_PROGRAMS =                  \
        inception

AM_CXXFLAGS =                      \
        $(RR_CXXFLAGS)             \
        $(CODE_COVERAGE_CXXFLAGS)  \

AM_CFLAGS =                        \
        $(RR_CFLAGS)               \
        $(CODE_COVERAGE_CFLAGS)

AM_CPPFLAGS =                      \
        $(RR_CPPFLAGS)             \
        $(CODE_COVERAGE_CPPFLAGS)  \
        -I../common/

AM_LDFLAGS =                       \
        $(RR_LIBS)                 \
        $(CODE_COVERAGE_LIBS)      \
        $(top_builddir)/r2i/libr2inference-@RR_PACKAGE_VERSION@.la

endif # ENABLE_EXAMPLES
