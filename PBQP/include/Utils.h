#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>
#include <vector>

//#define DEBUG

#ifdef DEBUG
#define DEBUG_EXPR(expr) expr
#else
#define DEBUG_EXPR(expr)
#endif

#define CUDA_CHECK(expr)                                                       \
  {                                                                            \
    auto MyErr = (expr);                                                       \
    if (MyErr != cudaSuccess) {                                                \
      printf("%s in %s at line %d\n", cudaGetErrorString(MyErr), __FILE__,     \
             __LINE__);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  }

namespace utils {

void printDeviceLimits(std::ostream &S);
void checkKernelsExec();

template <typename T1, typename T2>
__device__ __host__ unsigned ceilDiv(T1 Lhs, T2 Rhs) {
  auto LhsF = static_cast<float>(Lhs);
  auto RhsF = static_cast<float>(Rhs);
  return ceil(LhsF / RhsF);
}

template <typename T1, typename T2> struct Pair {
  T1 First;
  T2 Second;
};

template <typename It> class IteratorRange {
  It Begin;
  It End;

public:
  IteratorRange(It Begin, It End) : Begin{Begin}, End{End} {}

  It begin() { return Begin; }
  It end() { return End; }
};

template <typename It> IteratorRange<It> makeRange(It Begin, It End) {
  return IteratorRange<It>{Begin, End};
}

template <typename It> void print(It Beg, It End, std::ostream &S) {
  for (auto I : makeRange(Beg, End))
    S << I << " ";
  S << "\n";
}

template <typename T>
void printMatrix(const T &Matrix, std::ostream &S) {
  for (size_t i = 0; i < Matrix.h(); ++i) {
    for (size_t j = 0; j < Matrix.w(); ++j)
      S << Matrix[i][j] << " ";
    S << "\n";
  }
}

template <typename T>
std::set<T> sub(const std::set<size_t> &Lhs, const std::set<T> &Rhs) {
  auto Ans = Lhs;
  for (auto I : Rhs)
    Ans.erase(I);
  return Ans;
}

void reportFatalError(const std::string &Msg);

struct CLOption final {
  enum class Type { Flag, String };

private:
  std::string Name;
  Type Type;

  bool matchName(const std::string &ArgName) const {
    if (ArgName == Name || ArgName == "-" + Name || ArgName == "--" + Name)
      return true;
    return false;
  }

  template <typename It>
  std::pair<It, std::string> getArg(It Beg, It End) const {
    if (Beg == End && Type == Type::String)
      utils::reportFatalError("Empty option [" + Name + "]");
    if (Beg == End && Type == Type::Flag)
      return {Beg, "true"};
    auto Arg = *Beg;
    if (Type == Type::Flag) {
      if (Arg == "false" || Arg == "true")
        return {++Beg, Arg};
      return {Beg, "true"};
    }
    if (Type == Type::String)
      return {++Beg, Arg};
    reportFatalError("Unreachable");
    return {Beg, ""};
  }

public:
  CLOption(const std::string &Name, enum Type Type) : Name{Name}, Type{Type} {}

  // If option has been matched returns iterator and info
  template <typename It>
  std::pair<It, std::string> match(It Beg, It End) const {
    if (Beg == End)
      return {Beg, ""};
    if (!matchName(*Beg))
      return {Beg, ""};
    ++Beg;
    return getArg(Beg, End);
  }

  const std::string &getName() const { return Name; }
};

class CLParser final {
  std::vector<std::string> Args;
  std::vector<CLOption> Options;
  std::vector<std::pair<CLOption, std::string>> ParsedOptions;

public:
  CLParser(int Argc, char **Argv) : Args(Argv + 1, Argv + Argc) {}

  void addOption(const std::string &Name, enum CLOption::Type TypeVal) {
    Options.emplace_back(Name, TypeVal);
  }

  bool parseOptions() {
    auto ArgsEnd = Args.end();
    for (auto It = Args.begin(); It != ArgsEnd;) {
      auto PrevIt = It;
      std::for_each(Options.begin(), Options.end(), [&](const CLOption &Opt) {
        auto [NewIt, Val] = Opt.match(It, ArgsEnd);
        It = NewIt;
        if (Val != "")
          ParsedOptions.emplace_back(Opt, std::move(Val));
      });
      if (PrevIt == It)
        return false;
    }
    return true;
  }

  std::string getOption(const std::string &Name) {
    auto It = std::find_if(ParsedOptions.begin(), ParsedOptions.end(),
                           [&Name](const auto &Opt) {
                             if (Name == Opt.first.getName())
                               return true;
                             return false;
                           });
    if (It != ParsedOptions.end())
      return It->second;
    return "";
  }
};

template <typename T> size_t to_milliseconds(const T &Clck) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clck).count();
}

template <typename T> size_t to_microseconds(const T &Clck) {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clck).count();
}

bool isEqual(double Lhs, double Rhs);

} // namespace utils