# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_NCSDK

noinst_LTLIBRARIES = libncsdk.la

# Define a custom area for our headers
ncsdkincludedir = $(includedir)/r2inference-$(RR_PACKAGE_VERSION)/r2i/ncsdk

libncsdk_la_SOURCES =			\
	parameters.cc			\
	model.cc			\
	loader.cc			\
	engine.cc			\
	frame.cc			\
	prediction.cc			\
	statuscodes.cc

libncsdk_la_CPPFLAGS = 			\
	$(RR_CPPFLAGS)			\
	$(CODE_COVERAGE_CPPFLAGS)

libncsdk_la_CFLAGS = 			\
	$(RR_CFLAGS)			\
	$(NCSDK_CFLAGS)			\
	$(CODE_COVERAGE_CFLAGS)

libncsdk_la_CXXFLAGS = 			\
	$(RR_CXXFLAGS)			\
	$(NCSDK_CFLAGS)			\
	$(CODE_COVERAGE_CXXFLAGS)

libncsdk_la_LIBADD = 			\
	$(RR_LIBS)			\
	$(NCSDK_LIBS)			\
	$(CODE_COVERAGE_LIBS)

ncsdkinclude_HEADERS =			\
	parameters.h			\
	model.h				\
	loader.h			\
	engine.h 			\
	prediction.h			\
	frame.h

endif # HAVE_NCSDK
