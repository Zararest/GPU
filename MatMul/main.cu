#include <chrono>
#include <iostream>




void withoutShared() {

}

void withShared() {

}

int main() {
  auto Start = std::chrono::steady_clock::now();
  withoutShared();
  auto End = std::chrono::steady_clock::now();
  auto Duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  std::cout << "Without shared: " << Duration.count() << std::endl;

  Start = std::chrono::steady_clock::now();
  withShared();
  End = std::chrono::steady_clock::now();
  Duration = std::chrono::duration_cast<std::chrono::milliseconds>(End - Start);
  std::cout << "With shared: " << Duration.count() << std::endl;
}