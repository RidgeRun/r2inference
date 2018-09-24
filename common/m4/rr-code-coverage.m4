# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.
#
# Allows the user to perform a code coverage analysis in the code
#
# RR_ENABLE_CODE_COVERAGE
#
# This macro installs a configure option to enable/disable support for code
# coverage.
#
# This macro will set the following makefile variables:
#   CODE_COVERAGE_CPPFLAGS
#   CODE_COVERAGE_CXXFLAGS
#   CODE_COVERAGE_CFLAGS
#   CODE_COVERAGE_LIBS
# 
# This variables should be used to build tests and libraries in the project.
#
# Additionally, RR_CODE_COVERAGE_RULES will be defined. Typically, the top
# level makefile will expand it using @RR_CODE_COVERAGE_RULES@
#

AC_DEFUN([RR_ENABLE_CODE_COVERAGE],[
  AX_CODE_COVERAGE

  RR_CODE_COVERAGE_RULES="$CODE_COVERAGE_RULES"
  AC_SUBST(RR_CODE_COVERAGE_RULES)
  AM_SUBST_NOTMAKE(RR_CODE_COVERAGE_RULES)

  AM_CONDITIONAL([ENABLE_CODE_COVERAGE], [test x$enable_code_coverage = xyes])
])
