dnl RR_ENABLE_EVAL

dnl ===========================================================================
dnl RR_ENABLE_EVAL
dnl
dnl Installs a configuration option to enable/disable evaluation mode. If
dnl enabled, an EVAL macro will be defined to be used in the code. 
dnl
dnl ===========================================================================
AC_DEFUN([RR_ENABLE_EVAL],
[
    dnl Allow mainainer to build eval version
    AC_ARG_ENABLE(eval,[
AS_HELP_STRING([--enable-eval], [Build the evaluation version of the plug-in (Disabled by default)])
    ])
    if test x$enable_eval = xyes; then
        AC_MSG_NOTICE([Building the EVAL version of the plug-in])
        AC_DEFINE(EVAL, 1, [Define this to build the evaluation version of the plug-in])
    fi
])
