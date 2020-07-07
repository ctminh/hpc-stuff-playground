#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

// path to find the dataset
const char *datafile = "../logfile.txt";

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

    // read file
    std::ifstream infile(datafile);
    std::string line;
    int idx, arg_num, size;
    double freq, exetime;
    char c1, c2, c3, c4;
    // while (std::getline(infile, line))
    while ((infile >> idx >> c1 >> arg_num >> c2 >> size >> c3 >> freq >> c4 >> exetime) && (c1 == c2 == c3 == c4 == ','))
    {
        std::cout << idx << std::endl;
    }

}