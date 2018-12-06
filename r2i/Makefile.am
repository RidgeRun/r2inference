# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

SUBDIRS =

if HAVE_NCSDK
SUBDIRS += ncsdk
endif

if HAVE_TENSORFLOW
SUBDIRS += tensorflow
endif

DIST_SUBDIRS = 		\
	ncsdk		\
	tensorflow

lib_LTLIBRARIES = libr2inference-@RR_PACKAGE_VERSION@.la

# Define a custom area for our headers
r2iincludedir = @R2IINCLUDEDIR@/r2i

libr2inference_@RR_PACKAGE_VERSION@_la_SOURCES =	\
	iframeworkfactory.cc				\
	runtimeerror.cc					\
	imageformat.cc

libr2inference_@RR_PACKAGE_VERSION@_la_CPPFLAGS = 	\
	$(RR_CPPFLAGS)					\
	$(CODE_COVERAGE_CPPFLAGS)

libr2inference_@RR_PACKAGE_VERSION@_la_CFLAGS = 	\
	$(RR_CFLAGS)					\
	$(CODE_COVERAGE_CFLAGS)

libr2inference_@RR_PACKAGE_VERSION@_la_CXXFLAGS = 	\
	$(RR_CXXFLAGS)					\
	$(CODE_COVERAGE_CXXFLAGS)

libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD = 	\
	$(RR_LIBS)					\
	$(CODE_COVERAGE_LIBS)

r2iinclude_HEADERS =		\
	frameworkmeta.h		\
	frameworks.h		\
	iengine.h		\
	iframe.h		\
	iframeworkfactory.h	\
	iloader.h		\
	imageformat.h		\
	imodel.h		\
	iparameters.h		\
	iprediction.h		\
	parametermeta.h		\
	r2i.h			\
	runtimeerror.h


if HAVE_NCSDK
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD += 	\
	$(top_builddir)/r2i/ncsdk/libncsdk.la
endif

if HAVE_TENSORFLOW
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD += 	\
	$(top_builddir)/r2i/tensorflow/libtensorflow.la
endif
