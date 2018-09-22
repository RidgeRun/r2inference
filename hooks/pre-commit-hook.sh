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

# Mainly based on GStreamer pre-commit hook

LOGGER_PROJECT_TAG="PRE COMMIT"; . common/logger.sh

indent_linter="common/linters/indentation.sh"
tests_linter="common/linters/tests.sh"

main()
{
  if [ -z "${DISABLE_LINTERS}" ] ; then
    $indent_linter || exit 1
    $tests_linter || exit 1
  fi
}

main $@
