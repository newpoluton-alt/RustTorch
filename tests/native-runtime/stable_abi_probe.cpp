#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void check(AOTITorchError status, const char* operation) {
  if (status != AOTI_TORCH_SUCCESS) {
    throw std::runtime_error(operation);
  }
}

using Owner = std::unique_ptr<AtenTensorOpaque, decltype(&aoti_torch_delete_tensor_object)>;
}

int main() {
  try {
    int64_t size = 3, stride = 1;
    AtenTensorHandle raw = nullptr;
    auto dtype = aoti_torch_dtype_uint16();
    auto device = aoti_torch_device_type_meta();
    check(aoti_torch_empty_strided(1, &size, &stride, dtype, device, -1, &raw), "empty_strided");
    Owner tensor(raw, aoti_torch_delete_tensor_object);

    AtenTensorHandle alias = nullptr;
    check(aoti_torch_new_tensor_handle(tensor.get(), &alias), "new_tensor_handle");
    Owner retained(alias, aoti_torch_delete_tensor_object);
    tensor.reset();

    int32_t actual_dtype = 0, actual_device = 0;
    check(aoti_torch_get_dtype(retained.get(), &actual_dtype), "get_dtype");
    check(aoti_torch_get_device_type(retained.get(), &actual_device), "get_device_type");
    if (actual_dtype != dtype || actual_device != device) {
      throw std::runtime_error("retained handle has incorrect metadata");
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
