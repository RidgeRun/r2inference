/* Copyright (C) 2020 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#ifndef R2I_ONNXRT_PARAMETER_ACCESSORS_H
#define R2I_ONNXRT_PARAMETER_ACCESSORS_H

#include <r2i/runtimeerror.h>
#include <r2i/iparameters.h>

namespace r2i {
namespace onnxrt {

class Accessor {
 public:
  Accessor();
  virtual ~Accessor() {}
  virtual RuntimeError Set(IParameters *target) = 0;
  virtual RuntimeError Get(IParameters *target) = 0;
};

class IntAccessor : public Accessor {
 public:
  IntAccessor () : Accessor() {}
  int value;
};

class StringAccessor : public Accessor {
 public:
  StringAccessor () {}
  std::string value;
};

class LoggingLevelAccessor : public IntAccessor {
 public:
  LoggingLevelAccessor () {}
  RuntimeError Set (IParameters *target);
  RuntimeError Get (IParameters *target);
};

class IntraNumThreadsAccessor : public IntAccessor {
 public:
  IntraNumThreadsAccessor () {}
  RuntimeError Set (IParameters *target);
  RuntimeError Get (IParameters *target);
};

class GraphOptLevelAccessor : public IntAccessor {
 public:
  GraphOptLevelAccessor () {}
  RuntimeError Set (IParameters *target);
  RuntimeError Get (IParameters *target);
};

class LogIdAccessor : public StringAccessor {
 public:
  LogIdAccessor () {}
  RuntimeError Set (IParameters *target);
  RuntimeError Get (IParameters *target);
};

} // namespace onnxrt
} // namespace r2i

#endif //R2I_ONNXRT_PARAMETER_ACCESSORS_H
