#include <torch/torch.h>
#include <iostream>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>

// path to find the dataset
const char *datafile = "../logfile.txt";


// struct for building the model
struct RegressionNet:torch::nn::Module{
    // declare the model
    RegressionNet(int n_features, int n_hidden, int n_output){
        hidden1 = register_module("hidden1", torch::nn::Linear(n_features, n_hidden));
        hidden2 = register_module("hidden2", torch::nn::Linear(n_hidden, n_hidden));
        predict = register_module("predict", torch::nn::Linear(n_hidden, n_output));

        torch::nn::init::xavier_uniform_(hidden1->weight);
        torch::nn::init::zeros_(hidden1->bias);
        torch::nn::init::xavier_uniform_(hidden2->weight);
        torch::nn::init::zeros_(hidden2->bias);
        torch::nn::init::xavier_uniform_(predict->weight);
        torch::nn::init::zeros_(predict->bias);
    }

    // implement the net algorithm
    torch::Tensor forward(torch::Tensor &input){
        // x = torch.tanh(self.hidden(x))  # activation function for hidden layer
        // x = torch.relu(self.hidden1(x))  # activation function for hidden layer
        // x = self.predict(x)  # linear output
        auto x = torch::tanh(hidden1(input));
        x = torch::relu(hidden2(x));
        x = predict(x);
        return x;
    }

    // ???
    torch::nn::Linear hidden1{nullptr}, hidden2{nullptr}, predict{nullptr};
};

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
    std::vector<std::vector<float>> size_freq; // for training 2d-array
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

        size_freq.emplace_back();
        size_freq.back().emplace_back(size_val);
        size_freq.back().emplace_back(freq_val);
    }

    // check the vector
    for (int i = 0; i < idx.size(); i++){
        // std::cout << idx.at(i) << "\t|"
        //             << arg_num.at(i) << "\t|"
        //             << size.at(i) << "\t|"
        //             << freq.at(i) << "\t|"
        //             << exetime.at(i) << std::endl;
        
        std::cout << size_freq[i][0] << ", " << size_freq[i][1] << std::endl;
    }

    // transform vector to tensor data
    torch::Tensor size_tensor = torch::from_blob(size.data(), {1000,1}, torch::kInt);
    auto dataset_size = torch::data::datasets::TensorDataset(size_tensor);
    auto data_loader_size = torch::data::make_data_loader(dataset_size, 1); // 1 is batch_size



}