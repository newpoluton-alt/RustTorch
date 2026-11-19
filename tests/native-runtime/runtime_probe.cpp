#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <c10/core/StreamGuard.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/custom_function.h>

#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {
void check_views_and_types() {
  auto view = [] {
    auto owner = at::arange(6).reshape({2, 3});
    return owner.transpose(0, 1);
  }();
  TORCH_CHECK(!view.is_contiguous() && view[2][1].item<int64_t>() == 5);
  auto alias = view;
  view[2][1].fill_(19);
  TORCH_CHECK(alias[2][1].item<int64_t>() == 19);

  auto meta = at::empty({2, 3}, at::TensorOptions().device(at::kMeta));
  TORCH_CHECK(meta.is_meta() && meta.sizes() == at::IntArrayRef({2, 3}));
  auto uint16 = at::ones({4}, at::TensorOptions().dtype(at::kUInt16));
  TORCH_CHECK(uint16.scalar_type() == at::kUInt16);
  TORCH_CHECK(uint16.to(at::kLong).sum().item<int64_t>() == 4);
  TORCH_CHECK(c10::Device("xpu:0").type() == c10::DeviceType::XPU);
  if (!at::hasXPU()) {
    bool unavailable = false;
    try {
      at::ones({1}, at::TensorOptions().device("xpu:0"));
    } catch (const c10::Error&) {
      unavailable = true;
    }
    TORCH_CHECK(unavailable, "a runtime without XPU accepted an XPU allocation");
  }
}

void check_generator() {
  auto generator = at::detail::createCPUGenerator(243);
  at::randn({5}, generator);
  at::Tensor checkpoint;
  {
    std::lock_guard lock(generator.mutex());
    checkpoint = generator.get_state();
  }
  auto expected = at::randn({12}, generator);
  TORCH_CHECK(!expected.equal(at::randn({12}, generator)));
  {
    std::lock_guard lock(generator.mutex());
    generator.set_state(checkpoint);
  }
  TORCH_CHECK(expected.equal(at::randn({12}, generator)));
}

struct DualLevel {
  uint64_t id = torch::autograd::forward_ad::enter_dual_level();
  ~DualLevel() { torch::autograd::forward_ad::exit_dual_level(id); }
};

void check_forward_ad() {
  auto input = at::arange(6, at::kDouble).reshape({2, 3}).transpose(0, 1);
  auto tangent = at::full_like(input, 3);
  at::Tensor retained;
  try {
    DualLevel level;
    auto dual = at::_make_dual(input, tangent, level.id);
    retained = dual.square();
    auto [value, derivative] = at::_unpack_dual(retained, level.id);
    TORCH_CHECK(value.equal(input.square()) && derivative.equal(6 * input));
    throw std::runtime_error("end scope");
  } catch (const std::runtime_error& error) {
    // c10::Error is also an exception. Only the deliberate unwind is success.
    TORCH_CHECK(std::string(error.what()) == "end scope", error.what());
  }
  DualLevel next;
  TORCH_CHECK(!std::get<1>(at::_unpack_dual(retained, next.id)).defined());
}

class StraightThroughRound : public torch::autograd::Function<StraightThroughRound> {
 public:
  static at::Tensor forward(torch::autograd::AutogradContext*, at::Tensor input) {
    return input.round();
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext*, torch::autograd::variable_list output_gradients) {
    return {output_gradients.at(0)};
  }
};

void check_custom_backward() {
  auto x = at::tensor({1.3, 1.8}, at::kDouble).set_requires_grad(true);
  auto rounded = StraightThroughRound::apply(x);
  TORCH_CHECK(rounded.equal(at::tensor({1., 2.}, at::kDouble)));
  auto seed = at::tensor({-2., 3.}, at::kDouble);
  auto gradients = torch::autograd::grad({rounded}, {x}, {seed});
  TORCH_CHECK(gradients.at(0).equal(seed));

  auto handle = x.register_hook([](at::Tensor) { throw std::runtime_error("hook failed"); });
  bool rejected = false;
  try {
    torch::autograd::grad({StraightThroughRound::apply(x)}, {x}, {seed});
  } catch (const std::runtime_error& e) {
    rejected = std::string(e.what()) == "hook failed";
  }
  x.remove_hook(handle);
  TORCH_CHECK(rejected);
  TORCH_CHECK(torch::autograd::grad({x.square()}, {x}, {seed}).at(0).equal(2 * x * seed));
}

void check_stream(c10::Device device) {
  c10::impl::VirtualGuardImpl backend(device.type());
  auto previous = backend.getStream(device);
  auto selected = backend.getNewStream(device);
  try {
    c10::StreamGuard scope(selected);
    TORCH_CHECK(backend.getStream(device) == selected);
    auto sum = at::ones({3}, at::TensorOptions().device(device)).sum();
    TORCH_CHECK(sum.item<float>() == 3.f);
    throw std::runtime_error("end scope");
  } catch (const std::runtime_error& e) {
    TORCH_CHECK(std::string(e.what()) == "end scope", e.what());
  }
  TORCH_CHECK(backend.getStream(device) == previous);
  std::cout << "original stream " << previous.id() << ", selected " << selected.id() << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2) {
      check_stream(c10::Device(argv[1]));
      return 0;
    }
    TORCH_CHECK(argc == 1, "expected zero arguments or one accelerator device");
    check_views_and_types();
    check_generator();
    check_forward_ad();
    check_custom_backward();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
