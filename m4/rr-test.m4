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
# Perform a check for a feature and install property to disable it
#
# RR_ENABLE_TESTS
#
# This macro installs a configure option to enable/disable support for unit
# testing. If enabled, it checks for the appropriate installation of CppUTest.
#
# This macro will set the ENABLE_TESTS makefile conditional to use in
# the project. Additionally will set TESTS_CFLAGS and TESTS_LIBS.
#

AC_DEFUN([RR_ENABLE_TESTS],[
  AC_ARG_ENABLE([tests],
    AS_HELP_STRING([--disable-tests], [Disable project testing]))

  AS_IF([test "x$enable_tests" != "xno"],[
    PKG_CHECK_MODULES([CPPUTEST],[cpputest],[
      AC_SUBST(CPPUTEST_CFLAGS, ["-include CppUTest/MemoryLeakDetectorNewMacros.h"])
      AC_SUBST(CPPUTEST_LIBS)
    ],[
    AC_MSG_ERROR([ 

No installation of CppUTest found. In Debian based systems you may install it by running:

   ~$ sudo apt-get install libcpputest-dev

Additionally, you may disable testing support by using "--disable-tests".

      ])
    ])
    
    AC_CHECK_PROGS(PYTHON_BIN,[python2.7 python],[
      AC_MSG_ERROR([No installation of Python found. In Debian based systems you may install it by running:])
      AC_MSG_ERROR([~$ sudo apt-get install python-minimal])
      AC_MSG_ERROR([Additionally, you may disable testing support by using "--disable-tests".])
      ])
      
    ],[
    AC_MSG_NOTICE([Testing support disabled!])
  ])
  AC_SUBST(TESTS_CFLAGS, [$CPPUTEST_CFLAGS])
  AC_SUBST(TESTS_LIBS, [$CPPUTEST_LIBS])
  AM_CONDITIONAL([ENABLE_TESTS], [test "x$enable_tests" != "xno"])
])
