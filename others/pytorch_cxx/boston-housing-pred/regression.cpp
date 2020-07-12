#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <string>

/*---------------- class of CSVrow -------------*/
class CSVRow{
    public:
        float operator[] (std::size_t index)
        {
            std::string &eg = m_data[index];
            return std::atof(eg.c_str());
        }

        std::size_t size() const
        {
            return m_data.size();
        }

        void readNextRow(std::istream &str)
        {
            std::string         line;
            std::getline(str, line);

            std::stringstream   lineStream(line);
		    std::string         cell;

            m_data.clear();

            while (std::getline(lineStream, cell, ','))
            {
                m_data.push_back(cell);
            }

            // This checks for a trailing comma with no data after it.
	        if (!lineStream && cell.empty()){
			    // If there was a trailing comma then add an empty element.
			    m_data.push_back("");
		    }
        }

    private:
        std::vector<std::string>    m_data;
};

/* ??? */
std::istream &operator >> (std::istream& str, CSVRow& data)
{
	data.readNextRow(str);
	return str;
}

/*---------------- normalize features ----------*/
std::vector<float> normalize_feature(std::vector<float> feat){
    using ConstIter = std::vector<float>::const_iterator;
    ConstIter max_element;
    ConstIter min_element;
    std::tie(min_element, max_element) = std::minmax_element(std::begin(feat), std::end(feat));

    float extra = *max_element == *min_element ? 1.0 : 0.0;
    for (auto& val: feat){
		// max_element - min_element + 1 to avoid divide by zero error
		val = (val - *min_element) / (*max_element - *min_element + extra);
	}

	return feat;
}

/*---------------- create data -----------------*/
template<typename T>
std::vector<T> linspace(int start, int end, int length) {
	std::vector<T> vec;
	T diff = (end - start) / T(length);
	for (int i = 0; i < length; i++) {
		vec.push_back(start + diff * i);
	}
	return vec;
}

template<typename T>
std::pair<std::vector<T>, std::vector<T>> create_data()
{
    int64_t m = 4;      // slope
	int64_t c = 6;      // intercept
    int start = 0;
    int end = 11;
    int length = 91;
    std::vector<T> y = linspace<T>(start, end, length);
    std::vector<T> x = y;

    // source: https://stackoverflow.com/a/3885136, y = y * m
	// this multiplies the vector with a scalar 
    std::transform(y.begin(), y.end(), y.begin(), [m](long long val){return val * m;});

	// source: https://stackoverflow.com/a/4461466, y = y + c
	std::transform(y.begin(), y.end(), y.begin(), [c](long long val){return val + c;});

	// y = y + <random numbers>, there are total 91 numbers
	// y = y + random(91, 2), calculate 91 random numbers and multiply each by 2
	std::vector<T> random_vector = random<T>(91, 2);
	std::vector<T> vec_sum_y = add_two_vectors<T>(y, random_vector);

	return std::make_pair(x, vec_sum_y);
}


/*---------------- process data ----------------*/
std::pair<std::vector<float>, std::vector<float>> process_data(std::ifstream &file){
    std::vector<std::vector<float>> features;
    std::vector<float> label;

    CSVRow row;
    file >> row;    // ignore the first row
    while (file >> row) {
		features.emplace_back();
		for (std::size_t elem = 0; elem < row.size(); ++elem) {
			features.back().emplace_back(row[elem]);        // each row has 14 elements
		}
		features.back() = normalize_feature(features.back());
		
		// Push final column to label vector
		label.push_back(row[row.size()-1]);
	}

    // Flatten features vectors to 1D
	std::vector<float> inputs = features[0];

	int64_t total = std::accumulate(std::begin(features) + 1, std::end(features), 0UL, 
                    [](std::size_t s, std::vector<float> const& v){
                        return s + v.size();
                    });

    inputs.reserve(total);
	for (std::size_t i = 1; i < features.size(); i++) {
		inputs.insert(inputs.end(), features[i].begin(), features[i].end());
	}

    return std::make_pair(inputs, label);
}


/*---------------- regression model ------------*/
struct Net : torch::nn::Module {
    Net(int in_dim, int out_dim){
        fc1 = register_module("fc1", torch::nn::Linear(in_dim, 500));
        fc2 = register_module("fc2", torch::nn::Linear(500, 500));
        fc3 = register_module("fc3", torch::nn::Linear(500, 200));
        fc4 = register_module("fc4", torch::nn::Linear(200, out_dim));
    }

    torch::Tensor forward(torch::Tensor x){
        x = fc1->forward(x);
        x = fc2->forward(x);
        x = fc3->forward(x);
        x = fc4->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
};

struct SampleNet : torch::nn::Module{
    Net(int in_dim, int out_dim){
        fc1 = register_module("fc1", torch::nn::Linear(in_dim, out_dim));
    }

    torch::Tensor forward(torch::Tensor x){
        x = fc1->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr};
};



/*---------------- main function ---------------*/
int main(int argc, char **argv)
{
    // load/read the datafile
    std::ifstream file;
    std::string path_to_file = "./BostonHousing.csv";
    file.open(path_to_file, std::ios_base::in);

    // store data into a struct - pair
    std::pair<std::vector<float>, std::vector<float>> dataset = process_data(file);
    std::vector<float> train_inputs = dataset.first;
    std::vector<float> train_outputs = dataset.second;
    // try with generated data
    std::vector<float> inputs, outputs;
    std::tie(inputs, outputs) = create_data<float>();

    // Phase1: data transforming
    auto train_outputs_tensor = torch::from_blob(train_outputs.data(), {int(train_outputs.size()), 1});
    auto train_inputs_tensor = torch::from_blob(train_inputs.data(), {int(train_outputs.size()), int(train_inputs.size()/train_outputs.size())});
    // try with generated data
    auto outputs_tensor = torch::from_blob(outputs.data(), { 91, 1});
    auto inputs_tensor = torch::from_blob(inputs.data(), { 91, 1});

    // Phase2: create the network
    // printf("Check: passed 1! input_size = %d\n", int(train_inputs_tensor.sizes()[1])-1);
    // auto net = std::make_shared<Net>(int(train_inputs_tensor.sizes()[1] - 1), 1);
    // printf("Check: passed 2! \n");
    // torch::optim::SGD optimizer(net->parameters(), 0.001);
    // printf("Check: passed 3! \n");

    // try with generated data
    auto net = std::make_shared<SampleNet>(1, 1);
    torch::optim::SGD optimizer(net->parameters(), 0.012);


    // Phase3: train and print loss
    std::size_t n_epochs = 100;
    for (std::size_t epoch = 1; epoch <= n_epochs; epoch++){
        auto out  = net->forward(inputs_tensor);
        optimizer.zero_grad();

        auto loss = torch::mse_loss(out, outputs_tensor);
        float loss_val = loss.item<float>();

        loss.backward();
        optimizer.step();

        std::cout << "Loss: " << loss_val << std::endl;
    }

    return 0;

}