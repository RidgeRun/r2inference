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

LOGGER_PROJECT_TAG="LINT TESTS"; . common/logger.sh

test_driver="common/rr-test-driver.py"

run_tests()
{
    ninja -C build test
    ret=$?
    
    return $ret
}

main()
{
    log_info "Running project tests"
    
    run_tests
    ret=$?

    if test $ret -eq 0; then
        log_info "Tests passing correctly!"
    else
        log_error "Some tests are failing :("
    fi
    
    exit $ret
}

main $@
