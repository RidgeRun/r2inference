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

# Logger util for build scripts. Define the following variables before including
# this script:
# LOGGER_PROJECT_TAG: The name of the project under which to log messages. Mandatory.
# LOGGER_DATE_FMT: The date format used to log time. Defaults to '%T'
# LOGGER_SILENT: Whether or not to print output. Defaults to false

logger_date_fmt_default='%T'
logger_silent_default=false

if test -z "$LOGGER_PROJECT_TAG"; then
    echo "Please define LOGGER_PROJECT_TAG before including the logger script" >&2
    exit 1
fi

if test -z "$LOGGER_DATE_FMT"; then
    LOGGER_DATE_FMT=$logger_date_fmt_default
fi

if test -z "$LOGGER_SILENT"; then
    LOGGER_SILENT=$logger_silent_default
fi

log()
{
    level=$1
    shift
    msg=$@

    if ! test $LOGGER_SILENT = 1; then
        printf "[%s][%s][%s]\t%s\n" "$LOGGER_PROJECT_TAG" "`date +"$LOGGER_DATE_FMT"`" "$level" "$msg"
    fi
}

log_info()
{
    msg=$@
    
    log "INFO" $msg
}

log_warning()
{
    msg=$@

    log "WARN" $msg 1>&2
}

log_error()
{
    msg=$@

    log "ERROR" $msg 1>&2
}

# Small test, run as:
# sh -c "LOGGER_PROJECT_TAG="TEST"; . logger.sh; logger_test"
logger_test()
{
    log_info This is a test
    log_warning This is a warning
    log_error This is an error
}
