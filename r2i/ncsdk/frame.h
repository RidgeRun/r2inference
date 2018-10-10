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
#ifndef R2I_NCSDK_FRAME_H
#define R2I_NCSDK_FRAME_H

#include <r2i/iframe.h>
#include <r2i/runtimeerror.h>

namespace r2i {
namespace ncsdk {

class Frame: public IFrame {
 public:
  void *GetData () override;
  unsigned int GetSize () override;
  void SetData (void *graph_data);

 private:
  void *data;
  unsigned int graph_size;
};

}
}


#endif //R2I_NCSDK_FRAME_H
