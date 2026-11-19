#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 3, "expected rank and temporary rendezvous path");
    const int rank = std::stoi(argv[1]);
    auto store = c10::make_intrusive<c10d::FileStore>(argv[2], 2);
    auto options = c10d::ProcessGroupGloo::Options::create(std::chrono::seconds(15));
    options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
    auto group = c10::make_intrusive<c10d::ProcessGroupGloo>(store, rank, 2, options);
    std::vector<at::Tensor> values{at::scalar_tensor(rank + 1, at::kLong)};
    auto work = group->allreduce(values);
    TORCH_CHECK(work->wait(std::chrono::seconds(15)));
    TORCH_CHECK(values.at(0).item<int64_t>() == 3);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
