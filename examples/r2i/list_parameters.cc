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
      if (error.IsError ()) {
        return "error: " + error.GetDescription();
      } else {
        return std::to_string (value);
      }
    }

    case (r2i::ParameterMeta::Type::STRING): {
      std::string value (256, 0);
      error = params->Get (param.name, value);
      if (error.IsError ()) {
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

void ListFramework (r2i::FrameworkMeta &meta, const std::string graph) {
  r2i::RuntimeError error;

  std::cout << "Listing " << meta.name << " parameters" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << std::endl;

  auto factory = r2i::IFrameworkFactory::MakeFactory(meta.code, error);

  auto loader = factory->MakeLoader(error);

  auto model = loader->Load(graph, error);
  if (error.IsError ()) {
    std::cerr << "Unable to load model " << error << std::endl;
    return;
  }

  std::shared_ptr<r2i::IEngine> engine = factory->MakeEngine(error);

  error = engine->SetModel(model);
  if (error.IsError ()) {
    std::cerr << "Unable to set model " << error << std::endl;
    return;
  }

  error = engine->Start ();
  if (error.IsError ()) {
    std::cerr << "Unable to start engine " << error << std::endl;
    return;
  }

  std::shared_ptr<r2i::IParameters> params = factory->MakeParameters(error);
  error = params->Configure (engine, model);

  std::vector<r2i::ParameterMeta> desc;
  error = params->List (desc);

  for (auto &param : desc) {
    PrintParameter (params, param);
  }
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

  for (auto &meta : r2i::IFrameworkFactory::List (error)) {
    ListFramework (meta, graph);
  }

  return 0;
}

