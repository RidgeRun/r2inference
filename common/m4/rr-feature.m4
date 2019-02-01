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
# RR_CHECK_FEATURE_LIB(FEATURE-NAME, FEATURE-DESCRIPTION,
#                      LIBRARY, TEST-FUNCTION, HEADER,  OPTIONAL)
#
# This macro adds a command line argument to allow the user to enable
# or disable a feature, and if the feature is enabled, performs a supplied
# test to check if the feature is available.
#
# The test should define HAVE_<FEATURE-NAME> to "yes" or "no" depending
# on whether the feature is available.
#
# FEATURE-NAME          is the name of the feature, and should be in
#                       purely upper case characters.
# FEATURE-DESCRIPTION   is used to describe the feature in help text for
#                       the command line argument.
# LIBRARY               the library name for this feature.
# TEST-FUNCTION         is a function to test the library with.
# HEADER                the feature header to be included.
# OPTIONAL              [yes | no] Whether or not the feature is optional.

AC_DEFUN([RR_CHECK_FEATURE_LIB],[
FEATURE_NAME_LOWERCASE=translit([$1], A-Z, a-z)
AC_MSG_NOTICE(*** checking feature: $FEATURE_NAME_LOWERCASE ***)
RR_FEATURES_ALL="$RR_FEATURES_ALL $FEATURE_NAME_LOWERCASE"

AC_ARG_ENABLE(translit([$1], A-Z, a-z),
  [  ]builtin(format, --%-21s enable %s, enable-$FEATURE_NAME_LOWERCASE, [$2]),
  [ case "${enableval}" in
      yes) USE_[$1]=yes;;
      no) USE_[$1]=no;;
      *) AC_MSG_ERROR(bad value ${enableval} for --enable-$FEATURE_NAME_LOWERCASE) ;;
    esac],
  [ USE_$1=no]) # DEFAULT
PROP_NAME=--enable-$FEATURE_NAME_LOWERCASE
RR_FEATURES_ENABLE="$RR_FEATURES_ENABLE $PROP_NAME"

if test x$USE_[$1] = xyes; then
  RR_FEATURES_SELECTED="$RR_FEATURES_SELECTED $FEATURE_NAME_LOWERCASE"
  HAVE_[$1]=no
  AC_CHECK_LIB($3,
               $4,
               [HAVE_[$1]=yes],
               [HAVE_[$1]=no
               if test x[$6] = xno; then
               AC_MSG_ERROR([Couldn't find $3])
               fi ])
  AC_CHECK_HEADER($5,
               [HAVE_[$1]=yes],
               [HAVE_[$1]=no
               if test x[$6] = xno; then
               AC_MSG_ERROR([Couldn't find $3])
               fi ])
  
  if test x$HAVE_[$1] = xno; then
    RR_FEATURES_NO="$RR_FEATURES_NO $FEATURE_NAME_LOWERCASE"
    USE_[$1]=no
  else
    RR_FEATURES_YES="$RR_FEATURES_YES $FEATURE_NAME_LOWERCASE"
    AC_MSG_NOTICE(*** This feature will be used: [$2])
    AC_SUBST([$1_CFLAGS], [])
    AC_SUBST([$1_LIBS], [-l[$3]])
    AC_DEFINE_UNQUOTED([HAVE_$1], [1], [Support for $1 backend is enabled])
  fi
else
  HAVE_[$1]=no
  AC_MSG_NOTICE(*** This feature is disabled: [$2])
fi
AM_CONDITIONAL([HAVE_$1], [test x$HAVE_[$1] = xyes])
])


# Print  a summary of the features checked 
#
# RR_OUTPUT_FEATURES([HEADER])
#
# HEADER			is an optional header to print alongside the summary 

AC_DEFUN([RR_OUTPUT_FEATURES], [


if test "x$RR_FEATURES_SELECTED" = "x"; then
	printf "configure: *** Please select the features to use with at least one of the following options:\n"
	( for i in $RR_FEATURES_ENABLE; do printf '\t'$i'\n'; done ) | sort
	AC_MSG_ERROR(No features selected)
fi

printf "**************************************************\n"
printf ifelse([$1],,'\n','**********\t'[$1]'\t\t**********\n\n')

printf "*** Enabled features:\n\n"
( for i in $RR_FEATURES_SELECTED; do printf '\t'$i'\n'; done ) | sort
printf "\n"

printf "*** Disabled features:\n\n"
( for i in $RR_FEATURES_ALL; do
    case " $RR_FEATURES_SELECTED " in
      *\ $i\ *)
	;;
      *)
	printf '\t'$i'\n'
	;;
    esac
  done ) | sort
printf "\n"
printf "*** Enabled features that will be built:"
printf "$RR_FEATURES_YES\n" | sort
printf "\n"
printf "*** Enabled features that will NOT be built:"
printf "$RR_FEATURES_NO\n" | sort
printf "\n"
printf "**************************************************\n"
])
