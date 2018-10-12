# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

bin_PROGRAMS = example

example_CXXFLAGS = 				\
	$(RR_CXXFLAGS)

example_CPPFLAGS = 				\
	$(RR_CPPFLAGS)

example_LDFLAGS = 				\
	$(RR_LIBS)						\
	$(NCSDK_LIBS)					\
	$(top_builddir)/r2i/libr2inference-@RR_PACKAGE_VERSION@.la

example_SOURCES = 				\
	example.cc
