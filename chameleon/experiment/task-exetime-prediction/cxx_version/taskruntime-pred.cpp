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
    std::vector<int> idx;       // for index
    std::vector<int> arg_num;   // for num of arguments
    std::vector<int> size;      // for array of sizes
    std::vector<double> freq;   // for array of freq
    std::vector<double> exetime;    // for array of exectime
    int idx_val, arg_val, size_val; // get value when reading file
    double freq_val, exetime_val;   // get value when reading file
    char c1, c2, c3, c4;
    
    while ((infile >> idx_val >> c1 >> arg_val >> c2 >> size_val >> c3 >> freq_val >> c4 >> exetime_val))
    {
        // checking file-data
        // std::cout << idx_val << "|" << size_val << "|" << freq_val << "|" << exetime_val << std::endl;
        idx.push_back(idx_val);
        arg_num.push_back(arg_val);
        size.push_back(size_val);
        freq.push_back(freq_val);
        exetime.push_back(exetime_val);
    }

    // check the vector
    for (int i = 0; i < idx.size(); i++){
        std::cout << idx.at(i) << "\t|"
                    << arg_num.at(i) << "\t|"
                    << size.at(i) << "\t|"
                    << freq.at(i) << "\t|"
                    << exetime.at(i) << std::endl;
    }



}