#!/bin/sh
#
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

test=`echo $@ | sed -nr 's/^.*--test-name ([^ ]*).*$/\1/p'`
test_log_file=`echo $@ | sed -nr 's/^.*--log-file ([^ ]*).*$/\1/p'`
test_trs_file=`echo $@ | sed -nr 's/^.*--trs-file ([^ ]*).*$/\1/p'`
color_tests=`echo $@ | sed -nr 's/^.*--color-tests ([^ ]*).*$/\1/p'`

if [ -z $test ] || [ -z $test_log_file ] || [ -z $test_trs_file ]; then
    LOGGER_PROJECT_TAG="INDENT"; . common/logger.sh
    exit 255
fi

if [ x$color_tests = xyes ]; then
    red='[0;31m' # Red.
    grn='[0;32m' # Green.
    std='[m'     # No color.
else
    red=
    grn=
    std=
fi

echo -n "" > $test_log_file
echo -n "" > $test_trs_file

./$test -v 2>&1 | while read line; do
    echo $line | sed -nr "s/^(TEST\(.*\) - .*)$/${grn}PASS${std} - \1/p"
    echo $line | sed -nr "s/^(TEST\(.*\))$/${red}FAIL${std} - \1/p"

    echo $line | sed -nr 's/^TEST\((.*)\) - .*$/:test-result: PASS TEST(\1)/p' >> $test_trs_file
    echo $line | sed -nr 's/^TEST\((.*)\)$/:test-result: FAIL TEST(\1)/p' >> $test_trs_file
    
    echo $line >> $test_log_file
done
