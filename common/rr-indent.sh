#!/bin/sh
# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

. common/config.sh
LOGGER_PROJECT_TAG="INDENT"; . common/logger.sh

# Used for configurable pattern matching
shopt -s extglob

indent_tool="astyle"
indent_tool_version="3.0"
indent_tool_repo="https://sourceforge.net/projects/astyle/files/"

check_for_tool()
{
    ret=1
    if command -v $indent_tool >/dev/null 2>&1; then
      current_version="$($indent_tool --version | head -n1 | cut -d" " -f4)"
      if [ "$(printf '%s\n' "$indent_tool_version" "$current_version" | sort -V | head -n1)" = "$indent_tool_version" ] ; then
        ret=0
      fi
    fi

    if [ $ret -eq 1 ] ; then
      log_error "No \"$indent_tool\" code styler or insufficient version."
      log_error
      log_error "Please go to: $indent_tool_repo"
      log_error "and install latest version"
    fi

    return $ret;
}

load_defaults()
{
    INDENT_PROFILE_C=${INDENT_PROFILE_C-gstreamer}
    INDENT_PROFILE_H=${INDENT_PROFILE_H-gstreamer}
    INDENT_PROFILE_CXX=${INDENT_PROFILE_CXX-google}
    INDENT_PROFILE_HXX=${INDENT_PROFILE_HXX-google}

    INDENT_PATTERN_C=${INDENT_PATTERN_C-*.c}
    INDENT_PATTERN_H=${INDENT_PATTERN_H-*.h}
    INDENT_PATTERN_CXX=${INDENT_PATTERN_CXX-@(*.cc|*.cpp|*.c++)}
    INDENT_PATTERN_HXX=${INDENT_PATTERN_HXX-@(*.hpp|*.h++)}
}

get_parameters_from_profile()
{
    profile=$1

    log_info_stderr "Using profile: $profile"

    case $profile in
	google)
        echo "--style=google "
	;;

	gstreamer)
        echo " \
        --style=kr \
        --add-brackets \
        --align-pointer=middle \
        --align-reference=name \
        --lineend=linux \
        --pad-paren-out \
        --max-code-length=80 \
        --indent=spaces=2 \
        --indent-after-parens \
        --break-return-type "
	;;

	*)
        log_error "Unknown profile \"$profile\""
        ;;
    esac
}

get_parameters_from_extension()
{
    file=${INDENT_FORCE_EXTENSION-$1}

    if [ -z "$file" ]; then
	log_error "Can't deduct indent profile from empty file"
	exit 1
    fi

    case "$file" in
	$INDENT_PATTERN_C)
        indent_parameters=${INDENT_FLAGS_C-`get_parameters_from_profile $INDENT_PROFILE_C`}
	;;

	$INDENT_PATTERN_H)
        indent_parameters=${INDENT_FLAGS_H-`get_parameters_from_profile $INDENT_PROFILE_H`}
	;;

	$INDENT_PATTERN_CXX)
        indent_parameters=${INDENT_FLAGS_CXX-`get_parameters_from_profile $INDENT_PROFILE_CXX`}
	;;

	$INDENT_PATTERN_HXX)
        indent_parameters=${INDENT_FLAGS_HXX-`get_parameters_from_profile $INDENT_PROFILE_HXX`}
	;;

	*)
        log_warning "Unrecognized file extension \"$file\", you are on your own!"
        ;;
    esac
}

indent()
{
    params=$@

    if test -z "$params"; then
	get_parameters_from_extension
	$indent_tool $indent_parameters </dev/stdin
    else
	for param in `echo $params`; do
	    log_info "Indenting $param"
	    get_parameters_from_extension $param
	    $indent_tool $indent_parameters $params
	done
    fi
}

main()
{
    check_for_tool || exit 1
    load_defaults
    indent $@
}

main $@
