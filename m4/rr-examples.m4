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
# Allow the user to enable/disable the construction of examples
#
# RR_ENABLE_EXAMPLES
#
# This macro installs a configure option to enable/disable support for examples
# building.
#
# This macro will set the ENABLE_EXAMPLES makefile conditional to use in
# the project.
#
AC_DEFUN([RR_ENABLE_EXAMPLES],[
  AC_ARG_ENABLE([examples],
    AS_HELP_STRING([--disable-examples], [Disable project examples]))

  AS_IF([test "x$enable_examples" = "xno"],[
    AC_MSG_NOTICE([Examples disabled!])
  ])

  AM_CONDITIONAL([ENABLE_EXAMPLES], [test "x$enable_examples" != "xno"])
])
