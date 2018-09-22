# Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.

SUFFIXES=.cu .c

LTNVCOMPILE=$(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
	    $(LIBTOOLFLAGS) --mode=compile $(NVCC) $(CUDA_CFLAGS) \
	    $(EXTRA_CUDA_CFLAGS) -prefer-non-pic --compiler-options -fPIC
.cu.lo:
	$(LTNVCOMPILE) -c $< -o $@
