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

#ifndef R2I_ONNXRT_ACL_FRAMEWORK_FACTORY_H
#define R2I_ONNXRT_ACL_FRAMEWORK_FACTORY_H

#include <r2i/onnxrt/frameworkfactory.h>

namespace r2i {
namespace onnxrt_acl {

class FrameworkFactory : public r2i::onnxrt::FrameworkFactory {
 public:
  std::unique_ptr<r2i::IEngine> MakeEngine (RuntimeError &error) override;

  r2i::FrameworkMeta GetDescription (RuntimeError &error) override;
};

} // namespace onnxrt_acl
} // namespace r2k

#endif //R2I_ONNXRT_ACL_FRAMEWORK_FACTORY_H
