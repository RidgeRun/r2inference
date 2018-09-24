#!/usr/bin/python

from __future__ import print_function

import argparse
import glob
import subprocess
import sys
import xml.etree.ElementTree as ET


class bcolors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'


class MODE(object):
    MODE_SIMPLE = 0
    MODE_TRS = 1
    MODE_REPORT = 2


class testresults(object):

    print_template = '''TEST RESULTS:
=============
'''
    repr_template = 'Test Results({ngroups} groups, {ntests} total tests)'

    def __init__(self, filename=None):
        self.testgroups = {}
        if filename:
            self.parse(filename)

    def parse(self, filename):
        tree = ET.parse(filename)
        for child in tree.getroot():
            if child.tag == 'properties':
                self.properties = child
            elif child.tag == 'system-out':
                self.stdout = child
            elif child.tag == 'system-err':
                self.stderr = child
            elif child.tag == 'testcase':
                self.addtestcase(child)
            else:
                print("Found unexpected child ", child.tag, ", ignoring it.",
                      file=sys.stderr)

    def addtestcase(self, child):
        groupname = child.attrib['classname']
        self.checkgroup(groupname)
        self.testgroups[groupname].addtest(child)

    def checkgroup(self, groupname):
        if groupname not in self.testgroups:
            self.testgroups[groupname] = testgroup(groupname)

    def get_report(self, mode=MODE.MODE_SIMPLE, color=False):
        str_repr = ""
        if mode in [MODE.MODE_SIMPLE, MODE.MODE_REPORT]:
            str_repr += self.print_template
        for name, group in self.testgroups.items():
            str_repr += group.get_report(mode=mode, color=color)
        return str_repr

    def __repr__(self):
        totaltests = 0
        for name, group in self.testgroups.items():
            totaltests += group.count()
        return self.repr_template.format(ngroups=len(self.testgroups.keys()),
                                         ntests=totaltests)


class testgroup(object):

    print_template = '''\nTEST GROUP: {name}\n----------- {dashes}\n'''
    repr_template = 'Test Group({ntests} total tests)'

    def __init__(self, name):
        self.name = name
        self.tests = []

    def addtest(self, xmlelement):
        newtest = testcase(xmlelement, self)
        if newtest:
            self.tests.append(newtest)

    def count(self):
        return len(self.tests)

    def get_report(self, mode=MODE.MODE_SIMPLE, color=False):
        str_repr = ""
        if mode in [MODE.MODE_SIMPLE, MODE.MODE_REPORT]:
            str_repr += self.print_template.format(name=self.name,
                                                   dashes='-' * len(self.name))
        for test in self.tests:
            str_repr += test.get_report(mode=mode, color=color)
        return str_repr

    def __repr__(self):
        return self.repr_template.format(ntests=self.count())


class testcase(object):

    print_template = "TEST({group}, {name}) - {time} ms [{result}]\n"
    trs_template = ":test-result: {result} TEST({group}, {name})" + \
                   " - {time} ms\n"
    RESULT_PASS = 0
    RESULT_FAIL = 1
    RESULT_SKIP = 2

    def __init__(self, xmlelement, group):
        attrib = xmlelement.attrib
        self.group = group
        self.name = attrib['name']
        self.time = float(attrib['time'])

        # Assume that the test was successful
        self.result = self.RESULT_PASS
        self.failure = None

        for child in xmlelement:
            if child.tag == 'failure':
                self.result = self.RESULT_FAIL
                self.failure = child.attrib
            elif child.tag == 'skipped':
                self.result = self.RESULT_SKIP

    def get_report(self, mode=MODE.MODE_SIMPLE, color=False):
        if mode == MODE.MODE_TRS:
            templ = self.trs_template
        else:
            templ = self.print_template
        if self.result == self.RESULT_PASS:
            status = 'PASS'
            if mode is not MODE.MODE_TRS and color:
                status = bcolors.GREEN + status + bcolors.ENDC
        elif self.result == self.RESULT_FAIL:
            status = 'FAIL'
            if mode is not MODE.MODE_TRS and color:
                status = bcolors.RED + status + bcolors.ENDC
        elif self.result == self.RESULT_SKIP:
            status = 'SKIP'
            if mode is not MODE.MODE_TRS and color:
                status = bcolors.YELLOW + status + bcolors.ENDC

        str_repr = templ.format(group=self.group.name,
                                name=self.name,
                                time=int(self.time * 1000),
                                result=status)
        return str_repr

    def __repr__(self):
        return self.get_report()


class testrunner(object):

    def __init__(self, testname, singleprocess, logfile, trsfile):
        self.testname = testname
        self.logfile = logfile
        self.trsfile = trsfile
        self.singleprocess = singleprocess
        self.results = testresults(filename=None)

    def run(self):
        self.runtests()
        self.parse()
        self.saveresults()

    def runtests(self):
        with open(self.logfile, mode='w') as log_file:
            args = ['./{name}'.format(name=self.testname), '-v', '-ojunit'];

            if self.singleprocess:
                args.append (self.singleprocess)

            subprocess.call(args, stdout=log_file, stderr=subprocess.STDOUT)

    def parse(self):
        xmlfiles = glob.glob('cpputest_*.xml')
        for f in xmlfiles:
            self.results.parse(f)

    def saveresults(self):
        with open(self.trsfile, mode='w') as trs_file:
            trs_file.write(self.results.get_report(mode=MODE.MODE_TRS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='CppUTest.')
    parser.add_argument('--color-tests')
    parser.add_argument('--trs-file')
    parser.add_argument('--log-file')
    parser.add_argument('--test-name')
    parser.add_argument('--enable-hard-errors')
    parser.add_argument('--expect-failure')
    parser.add_argument('--single-process')
    parser.add_argument('cmd_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if len(args.cmd_args) > 1 and args.cmd_args[0] == '--':
        testfilename = args.cmd_args[1]
    else:
        testfilename = args.test_name

    if args.color_tests == 'yes':
        usecolor = True
    else:
        usecolor = False

    if args.single_process == 'yes':
        singleprocess = ''
    else:
        singleprocess = '-p'

    runner = testrunner(testfilename, singleprocess, args.log_file, args.trs_file)
    runner.run()
    print(runner.results.get_report(mode=MODE.MODE_SIMPLE, color=usecolor))
