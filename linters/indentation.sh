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

LOGGER_PROJECT_TAG="LINT INDNT"; . common/logger.sh

indent_tool="common/rr-indent.sh"
indent_extensions="-e \.c$ -e \.cpp$ -e \.cc$ -e \.C$ -e \.cxx$ -e \.c\+\+$"

check_file_indentation()
{
    original=$1
    indented=`mktemp /tmp/${original}.XXXXXX` || exit 1

    if test -z "$file"; then
        log_error "No file provided to lint"
        exit 1;
    fi

    if ! test -e "$file"; then
        log_error "Unable to lint $file: No such file or directory"
        exit 1;
    fi

    $indent_tool < $original > $indented || exit 1
    diff -u -p "${original}" "${indented}"
    ret=$?

    rm "${indented}"

    if test $ret -eq 0; then
        log_info "PASS"
    else
        log_error "FAIL"
    fi

    return $ret
}

check_project_indentation()
{
    for file in `git diff-index --cached --name-only HEAD --diff-filter=ACMR| grep $indent_extensions` ; do
	log_info "Checking style in $file"

        # to_commit is the temporary checkout. This makes sure we check against the
        # revision in the index (and not the checked out version).
        to_commit=`git checkout-index --temp ${file} | cut -f 1`

        check_file_indentation $to_commit
        r=$?
        rm "${to_commit}"

        if [ $r != 0 ] ; then
            log_error "================================================================================================="
            log_error " Code style error in: $file                                                                      "
            log_error "                                                                                                 "
            log_error " Please fix before committing. Don't forget to run git add before trying to commit again.        "
            log_error " If the whole file is to be committed, this should work (run from the top-level directory):      "
            log_error "                                                                                                 "
            log_error "   common/rr-indent.sh $file; git add $file; git commit"
            log_error "================================================================================================="
            exit 1
        fi
    done

}

main()
{
    log_info "Checking project style"

    check_project_indentation
    ret=$?

    log_info "Project styled appropriately!"

    exit $ret
}

main $@
