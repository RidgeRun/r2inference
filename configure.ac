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
AC_INIT([RidgeRun inference library],[0.1.0],[https://github.com/RidgeRun/r2inference/issues],[r2inference])

# Initialize automake.
AM_INIT_AUTOMAKE([1.14 no-dist-gzip dist-xz tar-ustar foreign])

# Initialize libtool
LT_INIT

# Checks for programs.
AC_PROG_LN_S

# Initialize our build utils
RR_INIT

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

# Output files, please keep in alphabetical order
AC_CONFIG_FILES([
Makefile
docs/Makefile
docs/api/Makefile
docs/uml/Makefile
examples/Makefile
r2i/Makefile
tests/Makefile
tests/acceptance/Makefile
tests/unit/Makefile
tests/unit/r2i/Makefile
])

AC_OUTPUT

