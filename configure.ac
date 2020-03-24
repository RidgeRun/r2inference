# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

# Initialize autoconf.
AC_PREREQ([2.69])
AC_INIT([RidgeRun inference library],[0.6.0],[https://github.com/RidgeRun/r2inference/issues],[r2inference])

# Initialize our build utils
RR_INIT

# Initialize automake.
AM_INIT_AUTOMAKE([1.14 no-dist-gzip dist-xz tar-ustar foreign])

# Initialize libtool
LT_INIT

# Checks for programs.
AC_PROG_LN_S

# Allow the user to fine-tune project build
RR_ENABLE_TESTS
RR_ENABLE_EXAMPLES
RR_ENABLE_DOCS([\${top_srcdir}/r2i])
RR_ENABLE_CODE_COVERAGE

# Grab the project version number
RR_VERSION

# Make sure we are trying to build the correct area
AC_CONFIG_SRCDIR([r2i/r2i.h])

# Keep in sync with top Makefile.am
AC_CONFIG_MACRO_DIRS([m4 common/m4])

# Generate a header with useful definitions
AC_CONFIG_HEADER([config.h])

# Check for NCSDK
RR_CHECK_FEATURE_LIB(NCSDK, NCSDK Installation,
                     mvnc, ncGlobalSetOption, mvnc.h, no)

RR_CHECK_FEATURE_LIB(TENSORFLOW, TensorFlow Installation,
                     tensorflow, TF_Version, tensorflow/c/c_api.h, no)

AC_LANG_PUSH([C++])
RR_CHECK_FEATURE_LIB(TFLITE, TensorFlow lite Installation,
                     tensorflow-lite, TfLiteTypeGetName, tensorflow/lite/model.h, no)

RR_CHECK_FEATURE_LIB(TENSORRT, TensorRT Installation,
                     nvinfer, createInferBuilder_INTERNAL, NvInfer.h, no)
AC_LANG_POP([C++])

AM_COND_IF([HAVE_TENSORRT], [RR_CHECK_CUDA], [])

# Our Include path, to be shared in multiple makefiles
AC_SUBST([R2IINCLUDEDIR],[$includedir/r2inference-$RR_PACKAGE_VERSION/])

# Output files, please keep in alphabetical order
AC_CONFIG_FILES([
Makefile
docs/Makefile
docs/api/Makefile
docs/uml/Makefile
examples/Makefile
examples/r2i/Makefile
examples/r2i/ncsdk/Makefile
examples/r2i/tensorflow/Makefile
examples/r2i/tensorrt/Makefile
examples/r2i/tflite/Makefile
r2i/Makefile
r2i/ncsdk/Makefile
r2i/tensorflow/Makefile
r2i/tensorrt/Makefile
r2i/tflite/Makefile
r2inference-${RR_PACKAGE_VERSION}.pc:r2inference.pc.in
tests/Makefile
tests/acceptance/Makefile
tests/unit/Makefile
tests/unit/r2i/Makefile
tests/unit/r2i/ncsdk/Makefile
tests/unit/r2i/tensorflow/Makefile
tests/unit/r2i/tensorrt/Makefile
tests/unit/r2i/tflite/Makefile
])

AC_OUTPUT

RR_OUTPUT_FEATURES([R2Inference])
