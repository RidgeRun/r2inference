# Copyright (C) 2018-2020 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

SUBDIRS =

if HAVE_EDGETPU
SUBDIRS += edgetpu
endif

if HAVE_NCSDK
SUBDIRS += ncsdk
endif

if HAVE_ONNXRT
SUBDIRS += onnxrt
endif

if HAVE_ONNXRT_ACL
SUBDIRS += onnxrt_acl
endif

if HAVE_ONNXRT_OPENVINO
SUBDIRS += onnxrt_openvino
endif

if HAVE_TENSORFLOW
SUBDIRS += tensorflow
endif

if HAVE_TFLITE
SUBDIRS += tflite
endif

if HAVE_TENSORRT
SUBDIRS += tensorrt
endif

DIST_SUBDIRS =          \
        edgetpu         \
        ncsdk           \
        onnxrt          \
        onnxrt_acl      \
        onnxrt_openvino \
        tensorflow      \
        tensorrt        \
        tflite

lib_LTLIBRARIES = libr2inference-@RR_PACKAGE_VERSION@.la

# Define a custom area for our headers
r2iincludedir = @R2IINCLUDEDIR@/r2i

libr2inference_@RR_PACKAGE_VERSION@_la_SOURCES =    \
        datatype.cc                                 \
        iframeworkfactory.cc                        \
        iloader.cc                                  \
        runtimeerror.cc                             \
        imageformat.cc

libr2inference_@RR_PACKAGE_VERSION@_la_CPPFLAGS =   \
        $(RR_CPPFLAGS)                              \
        $(CODE_COVERAGE_CPPFLAGS)

libr2inference_@RR_PACKAGE_VERSION@_la_CFLAGS =     \
        $(RR_CFLAGS)                                \
        $(CODE_COVERAGE_CFLAGS)

libr2inference_@RR_PACKAGE_VERSION@_la_CXXFLAGS =   \
        $(RR_CXXFLAGS)                              \
        $(CODE_COVERAGE_CXXFLAGS)

libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD =     \
        $(RR_LIBS)                                  \
        $(CODE_COVERAGE_LIBS)

r2iinclude_HEADERS =         \
        datatype.h           \
        frameworkmeta.h      \
        frameworks.h         \
        iengine.h            \
        iframe.h             \
        iframeworkfactory.h  \
        iloader.h            \
        imageformat.h        \
        imodel.h             \
        iparameters.h        \
        ipostprocessing.h    \
        iprediction.h        \
        ipreprocessing.h     \
        parametermeta.h      \
        r2i.h                \
        runtimeerror.h


if HAVE_EDGETPU
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD +=    \
        $(top_builddir)/r2i/edgetpu/libedgetpu.la
endif

if HAVE_NCSDK
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD +=    \
        $(top_builddir)/r2i/ncsdk/libncsdk.la
endif

if HAVE_ONNXRT
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD +=    \
        $(top_builddir)/r2i/onnxrt/libonnxrt.la
endif

if HAVE_ONNXRT_ACL
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD +=    \
        $(top_builddir)/r2i/onnxrt_acl/libonnxrt_acl.la
endif

if HAVE_ONNXRT_OPENVINO
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD +=    \
        $(top_builddir)/r2i/onnxrt_openvino/libonnxrt_openvino.la
endif

if HAVE_TENSORFLOW
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD +=    \
        $(top_builddir)/r2i/tensorflow/libtf.la
endif

if HAVE_TENSORRT
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD +=    \
        $(top_builddir)/r2i/tensorrt/libtensorrt.la
endif

if HAVE_TFLITE
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD +=    \
        $(top_builddir)/r2i/tflite/libtflite.la
endif
