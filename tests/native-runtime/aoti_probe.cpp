#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <iostream>

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc == 2, "expected a locally compiled x.square() model package");
    torch::inductor::AOTIModelPackageLoader model(argv[1]);
    auto input = at::tensor({-3.f, 2.f, 0.f});
    auto outputs = model.run({input});
    TORCH_CHECK(outputs.size() == 1 && outputs.front().equal(input.square()));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
