# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

if HAVE_TENSORFLOW

noinst_LTLIBRARIES = libtensorflow.la

# Define a custom area for our headers
tensorflowincludedir = @R2IINCLUDEDIR@/r2i/tensorflow

libtensorflow_la_SOURCES =		\
	engine.cc

libtensorflow_la_CPPFLAGS = 		\
	$(RR_CPPFLAGS)			\
	$(CODE_COVERAGE_CPPFLAGS)

libtensorflow_la_CFLAGS = 		\
	$(RR_CFLAGS)			\
	$(TENSORFLOW_CFLAGS)		\
	$(CODE_COVERAGE_CFLAGS)

libtensorflow_la_CXXFLAGS = 		\
	$(RR_CXXFLAGS)			\
	$(TENSORFLOW_CFLAGS)		\
	$(CODE_COVERAGE_CXXFLAGS)

libtensorflow_la_LIBADD = 		\
	$(RR_LIBS)			\
	$(TENSORFLOW_LIBS)		\
	$(CODE_COVERAGE_LIBS)

tensorflowinclude_HEADERS =


endif # HAVE_TENSORFLOW
