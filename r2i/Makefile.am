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

DIST_SUBDIRS = ncsdk

lib_LTLIBRARIES = libr2inference-@RR_PACKAGE_VERSION@.la

# Define a custom area for our headers
r2iincludedir = $(includedir)/r2inference-$(RR_PACKAGE_VERSION)/r2i

libr2inference_@RR_PACKAGE_VERSION@_la_SOURCES =	\
	r2i.cc						\
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

r2iinclude_HEADERS =					\
	r2i.h						\
	iframeworkfactory.h				\
	runtimeerror.h					\
	imageformat.h

if HAVE_NCSDK
libr2inference_@RR_PACKAGE_VERSION@_la_LIBADD += 	\
	$(top_builddir)/r2i/ncsdk/libncsdk.la
endif
