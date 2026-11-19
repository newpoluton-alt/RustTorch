#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 4, "expected rank, rendezvous path and scenario");
    const std::string scenario = argv[3];
    TORCH_CHECK(scenario == "sum" || scenario == "rank_loss");
    const int rank = std::stoi(argv[1]);
    auto store = c10::make_intrusive<c10d::FileStore>(argv[2], 2);
    auto options = c10d::ProcessGroupGloo::Options::create(std::chrono::seconds(15));
    options->devices.push_back(c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
    auto group = c10::make_intrusive<c10d::ProcessGroupGloo>(store, rank, 2, options);
    std::vector<at::Tensor> values{at::scalar_tensor(rank + 1, at::kLong)};
    auto work = group->allreduce(values);
    TORCH_CHECK(work->wait(std::chrono::seconds(15)));
    TORCH_CHECK(values.at(0).item<int64_t>() == 3);
    // Rank one exits after a proven first collective. Rank zero must be able
    // to finish its next wait with a failure and release the process group.
    if (scenario == "rank_loss" && rank == 0) {
      bool rejected = false;
      try {
        auto pending = group->allreduce(values);
        rejected = !pending->wait(std::chrono::seconds(15));
      } catch (const std::exception&) {
        rejected = true;
      }
      TORCH_CHECK(rejected, "all-reduce unexpectedly succeeded without its peer");
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
