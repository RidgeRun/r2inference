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
# Initializes generic flags to be used around the project, and installs
# some configure customization options
#
# RR_INIT
#
# This macro will set the following variables that should be used around the
# project
#     RR_CFLAGS
#     RR_CXXFLAGS
#     RR_CPPFLAGS
#     RR_LIBS

AC_DEFUN([RR_INIT_FLAGS],[

  AC_REQUIRE([AC_PROG_CC])
  AC_REQUIRE([AC_PROG_CXX])
  AC_REQUIRE([AC_PROG_CPP])

  AC_ARG_WITH([profile], AS_HELP_STRING([--with-profile=release|debug|lazy],
    [Specify the build profile to use:
      (release: enables all optimizations)
      (debug: disables all optimizations and enables debug symbols)
      (lazy: same configurations as debug, but doesn't treat warnings as errors)
    ]),
  [with_profile=$withval],[with_profile=debug])

  strict_flags="-Werror -Wall"
  release_flags="-O3"
  debug_flags="-O0 -ggdb3"

  case $with_profile in
    release )
      RR_CFLAGS="$strict_flags $release_flags "
      ;;
    debug )
      RR_CFLAGS="$strict_flags $debug_flags "
      ;;
    lazy )
      RR_CFLAGS="$debug_flags "
      ;;
    * )
      AC_MSG_ERROR([Please specif a valid profile to use: debug|release|lazy])
      ;;
  esac

  AC_MSG_NOTICE([Using profile: $with_profile])

  RR_CXXFLAGS="$RR_CFLAGS --std=c++11"
  RR_CPPFLAGS="-I\$(top_srcdir) "

  AC_SUBST(RR_CFLAGS)
  AC_SUBST(RR_CXXFLAGS)
  AC_SUBST(RR_CPPFLAGS)
  AC_SUBST(RR_LIBS)
])

AC_DEFUN([RR_INIT],[
  AC_MSG_NOTICE([Thanks for using RidgeRun build utils!])

  RR_INIT_FLAGS
])
