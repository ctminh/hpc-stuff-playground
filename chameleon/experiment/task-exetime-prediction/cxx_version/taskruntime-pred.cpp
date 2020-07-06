#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

// path to find the dataset
const char *dataset = '../logfile.txt'

// main function
auto main() -> int
{
    // create seed
    torch::manual_seed(1);

    // choose the device for training
    torch::DeviceType device_type;
    if (torch::cuda::is_available()){
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }


}