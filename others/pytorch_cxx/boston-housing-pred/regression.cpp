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

    // Phase1: data transforming
    auto train_inputs_tensor = torch::from_blob(train_inputs.data(), {int(train_outputs.size()), int(train_inputs.size()/train_outputs.size())});
    auto train_outputs_tensor = torch::from_blob(train_outputs.data(), {int(train_outputs.size()), 1});

    // Phase2: create the network
    auto net = std::make_shared<Net>(int(train_inputs_tensor.sizes()[1]), 1);
    torch::optim::SGD optimizer(net->parameters(), 0.001);

    // Phase3: train and print loss
    std::size_t n_epochs = 100;
    for (std::size_t epoch = 1; epoch <= n_epochs; epoch++){
        auto out  = net->forward(train_inputs_tensor);
        optimizer.zero_grad();

        auto loss = torch::mse_loss(out, train_outputs_tensor);
        float loss_val = loss.item<float>();

        loss.backward();
        optimizer.step();

        std::cout << "Loss: " << loss_val << std::endl;
    }

    return 0;

}