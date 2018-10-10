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


#include "r2i/ncsdk/frame.h"
#include "r2i/ncsdk/statuscodes.h"

namespace r2i {
namespace ncsdk {

void *Frame::GetData () {
  return this->data;
}

unsigned int Frame::GetSize () {
  return this->graph_size;
}
void Frame::SetData (void *data) {
  this->data = data;
}
}
}

