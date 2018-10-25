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

# This is an autogen base script, you can use it in your git project and add
# it Ridgerun build tools support. To include it in your project, simply do:
#
#			$ ln -s common/autogen.sh autogen.sh

LOGGER_PROJECT_TAG="AUTOGEN"; . common/logger.sh

git_dir=.git
config_file=configure.ac
project_tag=`autoconf -t 'AC_INIT:${;}*' | cut -d\; -f1`
pre_commit_hook="common/hooks/pre-commit-hook.sh"

run_sanity_checks()
{
    #Make sure we are in project root dir
    if  test ! -f $config_file ; then
        log_error "You are not in root project directory"
        exit 1
    fi

    #check for tools
    log_info "Checking for autoreconf"
    which "autoreconf" >/dev/null ||
    {
        log_error "not found! Please install the autoconf package."
        exit 1
    }
	
    log_info "Checking for automake"
    which "automake" >/dev/null ||
    {
        log_error "not found! Please install the automake package."
        exit 1
    }

    log_info "Checking for pkg-config"
    which "pkg-config" >/dev/null ||
    {
        log_error "not found! Please install the pkg-config package."
        exit 1
    }
}

install_git_hook()
{
    #checking for hook pre-installation
    if ! test -d $git_dir; then
        log_info "Skipping indent git hook installation: Not a git repo"
        return;
    fi

    if test \( -x $git_dir/hooks/pre-commit -o -L $git_dir/hooks/pre-commit \); then
        log_info "Skipping indent git hook, previous hook found."
        return;
    fi

    log_info "Installing indent git hook"
    ln -s ../../$pre_commit_hook $git_dir/hooks/pre-commit
}

run_autoreconf()
{
    log_info "Running autoreconf..."
    autoreconf -fiv ||
    {
        log_error "Failed to run autoreconf."
        exit 1
    }
}

run_configure()
{
    if test ! -z "$NOCONFIGURE"; then
        log_info "Skipping configure step, as requested"
        return;
    fi
    log_info "Running configure with parameters $@"
    ./configure $@ || 
    {
        log_error "Failed to run configure"
        exit 1
    }
    
    #log_info "what do you have in package? $PACKAGE."
    log_info "Now type 'make' to compile $project_tag."
}

main()
{
    run_sanity_checks
    install_git_hook
    run_autoreconf
    run_configure $@
}

main $@
