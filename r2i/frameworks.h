/* Copyright (C) 2018 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
*/

#ifndef R2I_FRAMEWORKS_H
#define R2I_FRAMEWORKS_H

namespace r2i {
/**
 * Numerical codes identifying supported frameworks. Not that not
 * all frameworks will be available at runtime. For example, some of
 * them may be disabled at configure by the user, automatically if
 * no development where found or if the system doesn't seem to have
 * to appropriate hardware.
 */
enum FrameworkCode {
  /**
   * Intel Movidius Neural Compute software developer kit
   */
  NCSDK,

  /**
   * Google's TensorFlow
   */
  TENSORFLOW,

  /**
   * Google's TensorFlow Lite
   */
  TFLITE,

  /**
   * Number of supported frameworks, mostly for testing purposes.
   */
  MAX_FRAMEWORK
};

} //namespace r2i

#endif //R2I_FRAMEWORKS
