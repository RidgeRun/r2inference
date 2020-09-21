# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

lib_LTLIBRARIES = top_sort_postprocessing.la

top_sort_postprocessing_la_SOURCES =           \
    top_sort_postprocessing.cc

top_sort_postprocessing_la_CPPFLAGS =          \
    $(RR_CPPFLAGS)                            \
    $(CODE_COVERAGE_CPPFLAGS)                 \
    -I../../examples/r2i/common

top_sort_postprocessing_la_CFLAGS =            \
    $(RR_CFLAGS)                              \
    $(CODE_COVERAGE_CFLAGS)

top_sort_postprocessing_la_CXXFLAGS =          \
    $(RR_CXXFLAGS)                            \
    $(CODE_COVERAGE_CXXFLAGS)

#Indicate libtool this is a module (dlopened library)
top_sort_postprocessing_la_LDFLAGS =           \
    -module -avoid-version -shared
