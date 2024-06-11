#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace SSA {

class Value {
  size_t ID;
  size_t Type;
};

class Instruction {
  Value Def;
  std::vector<Value> Use;
  std::string Name;
};

} // namespace SSA

namespace MIR {

class Register final {
  size_t ID;
  size_t Type;
};

class Instruction {
  Register Def;
  std::vector<Register> Use;
  std::string Name;

public:
};

class Spill final : public Instruction {};
class Fill final : public Instruction {};

} // namespace MIR
