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

LOGGER_PROJECT_TAG="INDENT"; . common/logger.sh

indent_tool="astyle"
indent_tool_version="3.0"
indent_tool_repo="https://sourceforge.net/projects/astyle/files/"
indent_parameters=" \
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

indent()
{
    params=$@

    if test -z "$params"; then
        $indent_tool $indent_parameters </dev/stdin
    else
        log_info "Indenting $params"
        $indent_tool $indent_parameters $params
    fi
}

main()
{
    check_for_tool || exit 1
    indent $@
}

main $@
