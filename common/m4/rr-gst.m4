 #Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
# All Rights Reserved.
#
# The contents of this software are proprietary and confidential to RidgeRun,
# LLC.  No part of this program may be photocopied, reproduced or translated
# into another programming language without prior written consent of
# RidgeRun, LLC.  The user is free to modify the source code after obtaining
# a software license from RidgeRun.  All source code changes must be provided
# back to RidgeRun without any encumbrance.
#
# Perform a check for a feature and install property to disable it
#
# RR_ENABLE_PLUGINDIR
#
#
#

AC_DEFUN([RR_GST_PLUGINDIR],[
  AC_ARG_WITH([plugindir],
    [AS_HELP_STRING([--with-plugindir], [Select dir for plugins])],
    [plugindir=$withval],
    [plugindir="\$(libdir)/gstreamer-1.0"])

    AC_SUBST(plugindir)
    GST_PLUGIN_LDFLAGS='-module -avoid-version -export-symbols-regex [_]*\(gst_\|Gst\|GST_\).*'
    AC_SUBST(GST_PLUGIN_LDFLAGS)
    AC_MSG_NOTICE(Setting GStreamer plugindir to: $plugindir)
])

