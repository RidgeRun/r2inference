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

#include <iostream>
#include <r2i/r2i.h>
#include <unordered_map>

const std::string TypeToString (r2i::ParameterMeta::Type type) {
  switch (type) {
    case (r2i::ParameterMeta::Type::INTEGER):
      return "integer";
    case (r2i::ParameterMeta::Type::STRING):
      return "string";
    default:
      return "unkown";
  }
}

const std::string FlagsToString (int flags) {
  const unsigned int max = 1 << (sizeof(int) * 8 - 1);
  std::string ret;

  for (unsigned int flag = 1; flag < max; flag <<= 1) {
    switch (flag & flags) {
      case (r2i::ParameterMeta::Flags::READ):
        ret += "read ";
        break;
      case (r2i::ParameterMeta::Flags::WRITE):
        ret += "write ";
        break;
      default:
        break;
    }
  }

  return ret;
}

const std::string GetCurrentValue (std::shared_ptr<r2i::IParameters> params,
                                   r2i::ParameterMeta &param) {
  r2i::RuntimeError error;

  switch (param.type) {
    case (r2i::ParameterMeta::Type::INTEGER): {
      int value;
      error = params->Get (param.name, value);
      if (r2i::RuntimeError::Code::EOK != error.GetCode ()) {
        return "error: " + error.GetDescription();
      } else {
        return std::to_string (value);
      }
    }

    case (r2i::ParameterMeta::Type::STRING): {
      std::string value (256, 0);
      error = params->Get (param.name, value);
      if (r2i::RuntimeError::Code::EOK != error.GetCode ()) {
        return "error: " + error.GetDescription();
      } else {
        return value;
      }
    }
    default:
      return "enkown";
  }
}

void PrintParameter (std::shared_ptr<r2i::IParameters> params,
                     r2i::ParameterMeta &param) {
  std::cout << "Name          : " << param.name << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << "Description   : " << param.description << std::endl;
  std::cout << "Flags         : " << FlagsToString (param.flags) << std::endl;
  std::cout << "Type          : " << TypeToString (param.type) << std::endl;
  std::cout << "Current Value : " << GetCurrentValue (params, param) << std::endl;
  std::cout << std::endl;
}

int main (int argc, char *argv[]) {
  r2i::RuntimeError error;
  std::string graph;

  if (argc <= 1) {
    std::cerr << "Usage: " << argv[0] << " GRAPH_PATH" << std::endl;
    return 1;
  } else {
    graph = argv[1];
  }

  auto factory = r2i::IFrameworkFactory::MakeFactory(
                   r2i::IFrameworkFactory::Code::NCSDK, error);

  auto loader = factory->MakeLoader(error);

  auto model = loader->Load(graph, error);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    std::cerr << "Unable to load model (" << error.GetCode () << "): " <<
              error.GetDescription () << std::endl;
    return 1;
  }

  std::shared_ptr<r2i::IEngine> engine = factory->MakeEngine(error);

  error = engine->SetModel(model);
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    std::cerr << "Unable to set model (" << error.GetCode () << "): " <<
              error.GetDescription () << std::endl;
    return 1;
  }

  error = engine->Start ();
  if (r2i::RuntimeError::Code::EOK != error.GetCode()) {
    std::cerr << "Unable to start engine (" << error.GetCode () << "): " <<
              error.GetDescription () << std::endl;
    return 1;
  }

  std::shared_ptr<r2i::IParameters> params = factory->MakeParameters(error);
  error = params->Configure (engine, model);

  std::vector<r2i::ParameterMeta> desc;
  error = params->List (desc);

  std::cout << "Listing NCSDK parameters" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << std::endl;

  for (auto &param : desc) {
    PrintParameter (params, param);
  }

  return 0;
}
