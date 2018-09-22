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
# Allows the user to exclude the documentation from being built
#
# RR_ENABLE_DOCS
#
# This macro installs a configure option to enable/disable support for docs
# being built. If enabled, it checks for the appropriate installation of doxygen.
#
# This macro will set the ENABLE_DOCS makefile conditional to use in
# the project.
#

AC_DEFUN([RR_ENABLE_DOCS],[
  AC_ARG_ENABLE([docs],
    AS_HELP_STRING([--disable-docs], [Disable documentation]))

  AS_IF([test "x$enable_docs" != "xno"],[
    AC_CHECK_PROGS(DOXYGEN_BIN,[doxygen],[
      AC_MSG_ERROR([No installation of Doxygen found. In Debian based systems you may install it by running:])
      AC_MSG_ERROR([~$ sudo apt-get install doxygen])
      AC_MSG_ERROR([Additionally, you may disable testing support by using "--disable-docs".])
      ])
      
    ],[
    AC_MSG_NOTICE([Documentation support disabled!])
  ])

  AM_CONDITIONAL([ENABLE_DOCS], [test "x$enable_docs" != "xno"])
])
