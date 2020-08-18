# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

R2IVERSION=0.0

CXXFLAGS:= -std=c++11 $(shell pkg-config --cflags r2inference-$(R2IVERSION)) -pthread $(shell pkg-config --cflags glib-2.0 gmodule-2.0)
LIBS:=$(shell pkg-config --libs r2inference-$(R2IVERSION)) -pthread

OUT=list-backends
SOURCES=list_backends.cc
OBJECTS=$(SOURCES:.cc=.o)

CXX=g++

.PHONY: clean

$(OUT): $(OBJECTS)
	@$(CXX) -o $@ $< $(LIBS)

%.o: %.cc
	@$(CXX) -c -o $@ $(CXXFLAGS) $< 

clean:
	@rm -f $(OBJECTS) $(OUT) *~
