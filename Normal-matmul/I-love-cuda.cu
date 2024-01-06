#include <iostream>

#define RUNTIME_ERROR

#ifdef STATIC_ASSERT_FAIL
template <typename T>
struct Matrix {};

template <typename T>
void foo(T M) {
  static_assert(false, "Not implemented");
}

template <typename T>
void foo(Matrix<T> M) {
  std::cout << "Ok" << std::endl;  
}
#endif

#ifdef RUNTIME_ERROR
struct DeviceMatrix {
  __device__
  void doTheThing() {}
};

struct HostMatrix {
  __host__
  void doTheThing() {}
};

template <typename T>
void foo() {
  T{}.doTheThing();
}

#endif

int main() {
#ifdef STATIC_ASSERT_FAIL
  foo(Matrix<int>{});
#endif

  //DeviceMatrix{}.doTheThing(); error: calling a __device__ function("doTheThing") from a __host__ function("main") is not allowed
  foo<HostMatrix>();    // Ok
  foo<DeviceMatrix>();  // Build - Ok, Runtime - fail
}
