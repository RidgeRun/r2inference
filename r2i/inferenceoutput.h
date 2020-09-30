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

#ifndef R2I_INFERENCE_OUTPUT_H
#define R2I_INFERENCE_OUTPUT_H

/**
 * R2Inference Namespace
 */
namespace r2i {

enum InferenceOutputType {
  CLASSIFICATION,
  OBJECT_DETECTION,
  UNKNOWN_TYPE
};

class InferenceOutput {
 public:
  InferenceOutput() : type(InferenceOutputType::UNKNOWN_TYPE) {};
  ~InferenceOutput() {};

  void SetType(InferenceOutputType type) {
    this->type = type;
  }
  InferenceOutputType GetType() {
    return this->type;
  }
 protected:
  InferenceOutputType type;
};


}

#endif // R2I_INFERENCE_OUTPUT_H
