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
# Parses package version to extract independent semantic version numbers
#
# RR_VERSION
#
# This macro will set the RR_VERSION_MAJOR, RR_VERSION_MINOR, RR_VERSION_MICRO
# and RR_VERSION_NANO. These will be available in the code via the config.h and
# in Makefiles
#
AC_DEFUN([RR_VERSION],[
  RR_VERSION_MAJOR=`echo AC_PACKAGE_VERSION | cut -d'.' -f1`
  RR_VERSION_MINOR=`echo AC_PACKAGE_VERSION | cut -d'.' -f2`
  RR_VERSION_MICRO=`echo AC_PACKAGE_VERSION | cut -d'.' -f3`
  RR_VERSION_NANO=`echo AC_PACKAGE_VERSION | cut -d'.' -f4`

  AC_SUBST(RR_VERSION_MAJOR)
  AC_SUBST(RR_VERSION_MINOR)
  AC_SUBST(RR_VERSION_MICRO)
  AC_SUBST(RR_VERSION_NANO)

  AC_DEFINE_UNQUOTED([RR_VERSION_MAJOR], [$RR_VERSION_MAJOR], [Major package version])
  AC_DEFINE_UNQUOTED([RR_VERSION_MINOR], [$RR_VERSION_MINOR], [Minor package version])
  AC_DEFINE_UNQUOTED([RR_VERSION_MICRO], [$RR_VERSION_MICRO], [Micro package version])
  AC_DEFINE_UNQUOTED([RR_VERSION_NANO], [$RR_VERSION_NANO], [Nano package version])

  RR_PACKAGE_VERSION=$RR_VERSION_MAJOR.0
  AC_SUBST([RR_PACKAGE_VERSION])
  AC_DEFINE_UNQUOTED([RR_PACKAGE_VERSION], [$RR_PACKAGE_VERSION], [Package version])
])
