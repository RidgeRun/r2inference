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
# Allows the user to exclude the documentation from being built
#
# RR_ENABLE_DOCS
#
# This macro installs a configure option to enable/disable support for docs
# being built. If enabled, it checks for the appropriate installation of doxygen.
#
# This macro will set the ENABLE_DOCS makefile conditional to use in
# the project.
#

AC_DEFUN([RR_ENABLE_DOCS],[
  AC_ARG_ENABLE([docs],
    AS_HELP_STRING([--disable-docs], [Disable documentation]))

  AS_IF([test "x$enable_docs" != "xno"],[
    ifdef([DX_INIT_DOXYGEN],[
      DX_HTML_FEATURE(ON)
      DX_CHM_FEATURE(OFF)
      DX_CHI_FEATURE(OFF)
      DX_MAN_FEATURE(OFF)
      DX_RTF_FEATURE(OFF)
      DX_XML_FEATURE(OFF)
      DX_PDF_FEATURE(OFF)
      DX_PS_FEATURE(OFF)

      define([DOXYFILE], m4_default([$2], [${top_srcdir}/common/Doxyfile]))
      define([SOURCEDIR], m4_default([$1], [\${top_srcdir}]))

      DX_INIT_DOXYGEN([AC_PACKAGE_TARNAME],[DOXYFILE],[out])
      DX_ENV_APPEND(SRCDIR, [SOURCEDIR])

      RR_DOCS_RULES="
$DX_RULES

RR_DOCS_CLEANFILES=\$(DX_CLEANFILES)
docs-run: doxygen-doc
"
      AC_SUBST(RR_DOCS_RULES)
      AM_SUBST_NOTMAKE(RR_DOCS_RULES)
    ],[
      AC_MSG_ERROR([No installation of Doxygen found. In Debian based systems you may install it by running:
      ~$ sudo apt-get install doxygen
Additionally, you may disable testing support by using "--disable-docs".])
    ])
  ],[
    AC_MSG_NOTICE([Documentation support disabled!])
  ])

  AM_CONDITIONAL([ENABLE_DOCS], [test "x$enable_docs" != "xno"])
])
