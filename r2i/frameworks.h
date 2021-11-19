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
 * Numerical codes identifying supported frameworks. Note that not
 * all frameworks will be available at runtime. For example, some of
 * them may be disabled at configure by the user, automatically if
 * no development where found or if the system doesn't seem to have
 * the appropriate hardware.
 */
enum FrameworkCode {

  /**
   * Coral from Google
   */
  CORAL,

  /**
   * Open Neural Network Exchange Runtime
   */
  ONNXRT,

  /**
   * Open Neural Network Exchange Runtime (ARM Compute Library support)
   */
  ONNXRT_ACL,

  /**
   * Open Neural Network Exchange Runtime (OpenVINO support)
   */
  ONNXRT_OPENVINO,

  /**
   * Google's TensorFlow
   */
  TENSORFLOW,

  /**
   * Google's TensorFlow Lite
   */
  TFLITE,

  /**
   * NVIDIA's TensorRT
   */
  TENSORRT,

  /**
   * Number of supported frameworks, mostly for testing purposes.
   */
  MAX_FRAMEWORK,

  /**
   * Android's NPU delegate
   */
  NNAPI
};

}  // namespace r2i

#endif  // R2I_FRAMEWORKS
