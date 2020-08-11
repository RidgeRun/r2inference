# Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_ONNXRT_ACL

noinst_LTLIBRARIES = libonnxrt_acl.la

# Define a custom area for our headers
onnxrt_aclincludedir = @R2IINCLUDEDIR@/r2i/onnxrt_acl

libonnxrt_acl_la_SOURCES =                  \
    engine.cc                               \
    frameworkfactory.cc 

libonnxrt_acl_la_CPPFLAGS =                 \
    $(RR_CPPFLAGS)                          \
    $(CODE_COVERAGE_CPPFLAGS)

libonnxrt_acl_la_CFLAGS =                   \
    $(RR_CFLAGS)                            \
    $(ONNXRT_CFLAGS)                        \
    $(ONNXRT_ACL_CFLAGS)                    \
    $(CODE_COVERAGE_CFLAGS)

libonnxrt_acl_la_CXXFLAGS =                 \
    $(RR_CXXFLAGS)                          \
    $(ONNXRT_CFLAGS)                        \
    $(ONNXRT_ACL_CFLAGS)                    \
    $(CODE_COVERAGE_CXXFLAGS)

libonnxrt_acl_la_LIBADD =                   \
    $(RR_LIBS)                              \
    $(ONNXRT_LIBS)                          \
    $(ONNXRT_ACL_LIBS)                      \
    $(CODE_COVERAGE_LIBS)

onnxrt_aclinclude_HEADERS =                 \
    engine.h                                \
    frameworkfactory.h

endif # HAVE_ONNXRT_ACL
