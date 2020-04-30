project('r2inference', ['c','cpp'],
  version : '0.6.0',
  meson_version : '>= 0.50',)

project_name = meson.project_name()
project_version = meson.project_version()
version_arr = project_version.split('.')
version_major = version_arr[0].to_int()
version_minor = version_arr[1].to_int()
version_micro = version_arr[2].to_int()
if version_arr.length() == 4
  version_nano = version_arr[3].to_int()
else
  version_nano = 0
endif
version_is_dev = version_minor % 2 == 1 and version_micro < 90

# Get an object returns describing a compiler
cc = meson.get_compiler('c')

# Find external dependencies
if get_option('enable-tensorflow')
  tensorflow = cc.find_library('tensorflow', required: true)
  tensorflow_dep = declare_dependency(dependencies: tensorflow, include_directories: '/usr/local/include/tensorflow/')
endif

if get_option('enable-tflite')
  tensorflow_lite = cc.find_library('tensorflow-lite', required: true)
  tensorflow_lite_dep = declare_dependency(dependencies: tensorflow_lite)
  thread_dep = dependency('threads')
endif

## Dependencies
# Define library dependencies for Tensorflow support
if get_option('enable-tensorflow')
  lib_tf_dep = [tensorflow_dep]
endif
# Define library dependencies for Tensorflow Lite support
if get_option('enable-tflite')
  lib_tflite_dep = [tensorflow_lite_dep, thread_dep]
endif

#Find dependencies for TESTS
lib_tests = dependency('cpputest', required: true)
tests_dep = declare_dependency(dependencies: lib_tests)

# Verify if profiling was enabled
if get_option('enable-profiling').enabled()
  profiler_dep = dependency('libprofiler')
  if(profiler_dep.found())
    message('Profiling enabled: Building examples with profiling support.')
    link_flags += ['-lprofiler']
    c_flags += ['-DPROFILING']
    # Update test test_deps to include profiler dependency
    test_deps += [profiler_dep]
  else
    error('MESON_FAIL gperftools profiling library not found.')
  endif
endif

# Define header directories
configinc = include_directories('.')
r2i_inc_dir = include_directories('r2i')

# Define compiler args and include directories
c_args = ['-DHAVE_CONFIG_H', '-std=c++11']

if get_option('enable-tensorflow')
  c_args += ['-DHAVE_TENSORFLOW']
endif

if get_option('enable-tflite')
  c_args += ['-DHAVE_TFLITE']
endif

# Define installation directories
prefix = get_option('prefix')
lib_install_dir = get_option('libdir')

# Verify if the warning flags are available in the compiler
# If the flags is availale for the compiler it wiil be used in all compiler
# invocations with the exception of compile tests.
warning_flags = [
  '-Wmissing-declarations',
  '-Wmissing-prototypes',
  '-Wredundant-decls',
  '-Wundef',
  '-Wwrite-strings',
  '-Wformat',
  '-Wformat-nonliteral',
  '-Wformat-security',
  '-Wold-style-definition',
  '-Winit-self',
  '-Wmissing-include-dirs',
  '-Waddress',
  '-Waggregate-return',
  '-Wno-multichar',
  '-Wdeclaration-after-statement',
  '-Wvla',
  '-Wpointer-arith',
]
foreach extra_arg : warning_flags
  if cc.has_argument (extra_arg)
    # Add flag to the compiler command line
    add_project_arguments([extra_arg], language: 'c')
  endif
endforeach

# Create an empty configuration object to set config.h information
cdata = configuration_data()

# Set config.h information
cdata.set_quoted('LOCALEDIR', join_paths(get_option('prefix'), get_option('localedir')))
cdata.set_quoted('LIBDIR', join_paths(get_option('prefix'), get_option('libdir')))
cdata.set_quoted('PACKAGE', project_name)
cdata.set_quoted('PACKAGE_NAME', project_name)
cdata.set_quoted('PACKAGE_STRING', '@0@ @1@'.format(project_name,project_version))
cdata.set_quoted('PACKAGE_TARNAME', project_name)
cdata.set_quoted('PACKAGE_BUGREPORT', ' https://github.com/RidgeRun/r2inference/issues')
cdata.set_quoted('PACKAGE_VERSION', project_version)
cdata.set_quoted('VERSION', project_version)

# These are only needed/used by the ABI tests
host_defines = [
  [ 'x86', 'HAVE_CPU_I386' ],
  [ 'x86_64', 'HAVE_CPU_X86_64' ],
  [ 'arm', 'HAVE_CPU_ARM' ],
  [ 'aarch64', 'HAVE_CPU_AARCH64' ],
  [ 'mips', 'HAVE_CPU_MIPS' ],
  [ 'powerpc', 'HAVE_CPU_PPC' ],
  [ 'powerpc64', 'HAVE_CPU_PPC64' ],
  [ 'alpha', 'HAVE_CPU_ALPHA' ],
  [ 'sparc', 'HAVE_CPU_SPARC' ],
  [ 'ia64', 'HAVE_CPU_IA64' ],
  [ 'hppa', 'HAVE_CPU_HPPA' ],
  [ 'm68k', 'HAVE_CPU_M68K' ],
  [ 's390', 'HAVE_CPU_S390' ],
]
foreach h : host_defines
  if h.get(0) == host_machine.cpu()
    cdata.set(h.get(1), 1)
  else
    cdata.set(h.get(1), false)
  endif
endforeach

cdata.set_quoted('HOST_CPU', host_machine.cpu())

# Verify if the specified header exists
check_headers = [
  'dlfcn.h',
  'inttypes.h',
  'memory.h',
  'poll.h',
  'stdint.h',
  'stdlib.h',
  'stdio_ext.h',
  'strings.h',
  'string.h',
  'sys/param.h',
  'sys/poll.h',
  'sys/prctl.h',
  'sys/socket.h',
  'sys/stat.h',
  'sys/times.h',
  'sys/time.h',
  'sys/types.h',
  'sys/utsname.h',
  'sys/wait.h',
  'ucontext.h',
  'unistd.h',
  'valgrind/valgrind.h',
  'sys/resource.h',
]

foreach h : check_headers
  if cc.has_header(h)
    define = 'HAVE_' + h.underscorify().to_upper()
    cdata.set(define, 1)
  endif
endforeach

# Gtk documentation
gnome = import('gnome')

# Imports pkgconfig module
pkgconfig = import('pkgconfig')

# Install git hooks
python3 = import('python').find_installation()
run_command(python3, '-c', 'import shutil; shutil.copy("hooks/pre-commit.hook", ".git/hooks/pre-commit")')

# Meson will generate a header file all the entries in the configuration data object
configure_file(output : 'config.h', configuration : cdata)

# Enter to each subdirectory and execute the meson.build
subdir('r2i')
subdir('tests')
subdir('examples')
subdir('docs')
