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
# RR_CHECK_CUDA_GPU
#
# Checks if the system has a GPU or not. If so, this code gets some valuable
# information from the output of the deviceQuery tool present in the CUDA samples
# folder. This macro also add the compute capability to CFLAGS.
#
# If a GPU exists, this function fills variables with important information:
#
# * DRIVER_VERSION:		CUDA driver version
# * RUNTIME_VERSION:	CUDA runtime version
# * GLOBAL_MEMORY:		Total amount of global memory
# * THREADS_BLOCK:		Maximum threads per block
# * THREADS_MULTIPROC:	Maximum threads per multiprocessor
# * CAPABILITY:			CUDA capability major/minor version number
#
# RR_CHECK_CUDA
#
# Checks for CUDA in the default or given PREFIX location and fill the flags
#
# Checks the existance of cuda compiler (nvcc), cuda library header and
# binary (cuda.h, libcuda.so). If something isn't found, fails straight
# away. Also checks for the system's ability to run cuInit, because some
# systems may want to compile for CUDA without execution support.
#
# --with-cuda=PREFIX    Sets the CUDA installation pathnd.
#                       Switches --with-platform to gpu.
#                       Default prefix is /usr/local/cuda
#
# --with-platform=[auto|cpu|gpu]  Sets the desired platform for compilation.
#                                 Default platform is auto.
#                                 "auto" platform will search for CUDA and use
#                                 GPU if the compilation requisites are present.
#
# Fills variables to compile CUDA: NVCC, CUDA_CFLAGS, CUDA_LIBS and CUDA_LIBDIR
# Defines CUDA compilation flag and makefile conditional: PLATFORM_IS_GPU
#
# Based on Building a CUDA Device Function Library With Autotools:
# https://tschoonj.github.io/blog/2014/05/10/building-a-cuda-device-function-library-with-autotools/

AC_DEFUN([RR_CHECK_CUDA_GPU], [
	check_path="$cuda_prefix/samples/1_Utilities/deviceQuery/"
	file_name="deviceQuery"

	if [[ -d "$check_path" ]]; then
		if [[ ! -x $check_path$file_name ]]; then
			AC_MSG_ERROR([CUDA deviceQuery is needed to detect your hardware specs and it does not seems to be installed on your system.
To compile it please run:
   ~$ sudo make
On your: '$check_path' directory.])
		fi
		RESULT=$($check_path$file_name | grep -oP "(?<=Result = )[[^ ]]+")
    	if [[ "$RESULT" == "PASS" ]]; then
			AC_MSG_NOTICE([GPU test result: $RESULT])
    		VALID_CUDA_EXE=yes

			DRIVER_VERSION=$($check_path$file_name | grep -oP "(?<=CUDA Driver Version = ).*?(?=,)")
			AC_MSG_NOTICE([Current CUDA driver version: $DRIVER_VERSION])

			RUNTIME_VERSION=$($check_path$file_name | grep -oP "(?<=CUDA Runtime Version = ).*?(?=,)")
			AC_MSG_NOTICE([Current CUDA runtime version: $RUNTIME_VERSION])

			sort_version=$(echo "$RUNTIME_VERSION $DRIVER_VERSION" | tr " " "\n" | sort -V)
			greatest_version=$(echo $sort_version | tr "\n" " " | cut -d " " -f2)
			if [[[ $greatest_version == $DRIVER_VERSION ]]] && [[[ $RUNTIME_VERSION > $DRIVER_VERSION ]]]; then
				AC_MSG_ERROR([The runtime version is lower than the driver version])
			else
				GLOBAL_MEMORY=$($check_path$file_name | grep "global memory" | grep -oP "\(\K[[^ bytes)]]+")
				AC_MSG_NOTICE([Total amount of global memory: $GLOBAL_MEMORY bytes])

				BLOCKS_WARP=$($check_path$file_name | grep "Warp size" | grep -oP "[[0-9]]+")
				AC_MSG_NOTICE([Number of blocks per warp: $BLOCKS_WARP])

			 	THREADS_MULTIPROC=$($check_path$file_name | grep "threads per multiprocessor" | grep -oP "[[0-9]]+")
				AC_MSG_NOTICE([Number of threads per multiprocessor: $THREADS_MULTIPROC])

				THREADS_BLOCK=$($check_path$file_name | grep "threads per block" | grep -oP "[[0-9]]+")
				AC_MSG_NOTICE([Number of threads per block: $THREADS_BLOCK])

				CAPABILITY=$($check_path$file_name | grep "Capability" | grep -oP "[[0-9]]+.[[0-9]]+" | tr -d .)
				AC_MSG_NOTICE([CUDA capability version number: $CAPABILITY])
				CUDA_CFLAGS+=" --gpu-architecture=compute_$CAPABILITY"
			fi
		else
			VALID_CUDA_EXE=no
			AC_MSG_NOTICE([GPU test result: $RESULT])
			AC_MSG_NOTICE([The system does not have GPU])
		fi
	fi
])

AC_DEFUN([RR_CHECK_CUDA], [

# Provide platform preference
AC_ARG_WITH([platform], AS_HELP_STRING([--with-platform],
            [Specify the platform to use: auto|cpu|gpu.]),
            [with_platform=$withval],[with_platform=auto])

case $with_platform in
    auto )
        ;;
    cpu )
        ;;
    gpu )
        ;;
    * )
        AC_MSG_ERROR([Please specify the platform to use: auto|cpu|gpu]) ;;
esac
AC_MSG_NOTICE([Setting platform preference to: "$with_platform"])

# Provide CUDA path
AC_ARG_WITH([cuda],
            AS_HELP_STRING([--with-cuda=PREFIX],[Prefix of your CUDA installation]),
            [cuda_prefix=$withval; with_cuda="yes"],
            [cuda_prefix="/usr/local/cuda"; with_cuda="no"])

# Setting the prefix to the default if only --with-cuda was given
if test "$with_cuda" == "yes"; then
  if test "$with_platform" != "gpu"; then
    with_platform=gpu
    AC_MSG_NOTICE([Switched platform to "$with_platform" because --with-cuda option was provided])
  fi
  if test "$withval" == "yes"; then
    cuda_prefix="/usr/local/cuda"
    AC_MSG_NOTICE([Setting the prefix to the default: "$cuda_prefix"])
  fi
fi

# If platform is set to cpu there is no need to check for cuda
if test "$with_platform" != "cpu"; then

  # Checking for nvcc
  if test "$with_cuda" == "no"; then
    AC_PATH_PROG([NVCC], [nvcc], [no], ["$cuda_prefix/bin:$PATH"])
  else
    AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
    AS_IF([test -x "$cuda_prefix/bin/nvcc"],
          [NVCC="$cuda_prefix/bin/nvcc"; AC_MSG_RESULT([found])],
          [NVCC=no; AC_MSG_RESULT([not found!])])
  fi
  if test "$NVCC" != "no"; then
    AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
    # We need to add the CUDA search directories for header and lib searches

    CUDA_CFLAGS=""

    # Saving the current flags
    ax_save_CFLAGS="${CFLAGS}"
    ax_save_LDFLAGS="${LDFLAGS}"

    VALID_CUDA_COMPILER=yes
    VALID_CUDA_EXE=yes

    # Announcing the new variables
    AC_SUBST([NVCC],[$cuda_prefix/bin/nvcc])
    AC_SUBST([CUDA_INCDIR],[$cuda_prefix/include])
    AC_CHECK_FILE([$cuda_prefix/lib64],[lib64_found=yes],[lib64_found=no])
    if test "x$lib64_found" = xno ; then
      AC_CHECK_FILE([$cuda_prefix/lib],[lib32_found=yes],[lib32_found=no])
      if test "x$lib32_found" = xyes ; then
        AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
      else
        AC_MSG_WARN([Couldn't find cuda lib directory])
        VALID_CUDA_COMPILER=no
      fi
    else
      AC_CHECK_SIZEOF([long])
      if test "x$ac_cv_sizeof_long" = "x8" ; then
        AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib64])
        CUDA_CFLAGS+=" -m64"
      elif test "x$ac_cv_sizeof_long" = "x4" ; then
        AC_CHECK_FILE([$cuda_prefix/lib32],[lib32_found=yes],[lib32_found=no])
        if test "x$lib32_found" = xyes ; then
          AC_SUBST([CUDA_LIBDIR],[$cuda_prefix/lib])
          CUDA_CFLAGS+=" -m32"
        else
          AC_MSG_WARN([Couldn't find cuda lib directory])
          VALID_CUDA_COMPILER=no
        fi
      else
        AC_MSG_ERROR([Could not determine size of long variable type])
      fi
    fi

    if test "x$VALID_CUDA_COMPILER" != xno ; then
      CUDA_CFLAGS+=" -I$cuda_prefix/include"
      CFLAGS="-I$cuda_prefix/include $CFLAGS"
      CUDA_LIBS="-L$CUDA_LIBDIR"
      LDFLAGS="$CUDA_LIBS $LDFLAGS"

      # And the header and the lib
      AC_CHECK_HEADER([cuda.h], [],
        AC_MSG_WARN([Couldn't find cuda.h])
        VALID_CUDA_COMPILER=no
        ,[#include <cuda.h>])
      if test "x$VALID_CUDA_COMPILER" != "xno" ; then
	RR_CHECK_CUDA_GPU
      fi
    fi
    # Returning to the original flags
    CFLAGS=${ax_save_CFLAGS}
    LDFLAGS=${ax_save_LDFLAGS}
  else
    AC_MSG_WARN([nvcc was not found in $cuda_prefix/bin])
    VALID_CUDA_COMPILER=no
  fi
fi
AC_SUBST(CUDA_CFLAGS, ["$CUDA_CFLAGS -rdc=false"])
AC_SUBST(CUDA_LIBS, ["-L$CUDA_LIBDIR -lcuda -lcudart"])

case $with_platform in
    auto )
      AS_IF([test "x$VALID_CUDA_COMPILER" == "xyes" && test "x$VALID_CUDA_EXE" == "xyes"],
            [AC_DEFINE([PLATFORM_IS_GPU], [1], [Platform is GPU.]) AC_DEFINE_UNQUOTED([THREADS], [$THREADS_MULTIPROC], [Max threads per block]) AC_DEFINE_UNQUOTED([BLOCKS], [$BLOCKS_WARP], [Max blocks per warp]) AC_MSG_NOTICE([Platform automatically set to GPU])],
            [AC_MSG_NOTICE([Platform automatically set to CPU])])
      AM_CONDITIONAL([PLATFORM_IS_GPU], [test "x$VALID_CUDA_COMPILER" == "xyes" && test "x$VALID_CUDA_EXE" == "xyes"])
      ;;
    gpu )
      if test "x$VALID_CUDA_COMPILER" == "xyes" ; then
        AS_IF([test "x$VALID_CUDA_EXE" == "xno"],
              [AC_MSG_WARN([This system can't run CUDA applications])])
        AC_DEFINE([PLATFORM_IS_GPU], [1], [Platform is GPU.]) 
        AC_DEFINE_UNQUOTED([THREADS], [$THREADS_MULTIPROC], [Max threads per block])
	AC_DEFINE_UNQUOTED([BLOCKS], [$BLOCKS_WARP], [Max blocks per warp])
        AM_CONDITIONAL([PLATFORM_IS_GPU], [true])
        AC_MSG_NOTICE([Platform successfully set to GPU])
      else
        AC_MSG_ERROR([Could not set platform to GPU.])
      fi
      ;;
    cpu )
      AM_CONDITIONAL([PLATFORM_IS_GPU], [false])
      AC_MSG_NOTICE([Platform successfully set to CPU])
      ;;
esac

])
