#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<float> inputs(0.5, 0.5);
  // inputs.push_back(torch::rand(2));
  std::cout << inputs[0] << ", " << inputs[1] << std::endl;

  // Execute the model and turn its output into a tensor.
  // at::Tensor output = module.forward(inputs).toTensor();
  // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  std::cout << "ok\n";
}