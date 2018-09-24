# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

# Keep in sync with configure.ac
ACLOCAL_AMFLAGS = -I m4 -I common/m4

SUBDIRS =	\
	r2i

if ENABLE_TESTS
SUBDIRS += tests
endif

if ENABLE_EXAMPLES
SUBDIRS += examples
endif

if ENABLE_DOCS
SUBDIRS += docs
endif

DIST_SUBDIRS =		\
	$(SUBDIRS)	\
	tests		\
	examples	\
	docs

@CODE_COVERAGE_RULES@
