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
#
LOGGER_PROJECT_TAG="CONFIG"; . common/logger.sh

config_file=.rrconfig

config_check_for_tools()
{
    which "realpath" >/dev/null ||
    {
        log_error "Please install the realpath package."
        exit 1
    }
}

config_find_config_file()
{
    RRCONFIG=""

    olddir=0
    curdir=`realpath .`

    # We can't go back further, no config file found
    while [ x$olddir != x$curdir ]; do

	test_config=${curdir}/${config_file}
	if [ -r "$test_config" ]; then
	    log_info_stderr "Loading project configuration from \"${test_config}\""
	    RRCONFIG=$test_config
	    break
	fi

	olddir=$curdir
	curdir=`realpath ${curdir}/..`
    done
}

config_process_shameful()
{
    if [ ! -z "$DISABLE_LINTERS" ]; then
	log_info_stderr "You disabled linters, shame on you! >:("
    fi
}

config_main()
{
    RRCONFIG=""

    config_check_for_tools
    config_find_config_file

    if [ -r "$RRCONFIG" ]; then
	source $RRCONFIG
    else
	log_info "No custom config found. Using defaults"
    fi

    config_process_shameful
}

config_main $@
