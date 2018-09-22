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

# Checks for programs.
AC_PROG_CXX
AC_PROG_CPP
AC_PROG_LN_S

# Keep in sync with top Makefile.am
AC_SUBST(ACLOCAL_AMFLAGS, "-I m4 -I common/m4")

# Allow the user to fine-tune project build
RR_ENABLE_TESTS
RR_ENABLE_EXAMPLES

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

