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
        printf("row size: %d\n", row.size());
		for (std::size_t loop = 0;loop < row.size(); ++loop) {
			features.back().emplace_back(row[loop]);
		}
		features.back() = normalize_feature(features.back());
		
		// Push final column to label vector
		label.push_back(row[row.size()-1]);
	}

    // check the data after reading
    // for (int i = 0; i < label.size(); i++){
    //     std::cout << label.at(i) << std::endl;
    // }

    // Flatten features vectors to 1D
	std::vector<float> inputs = features[0];
	int64_t total = std::accumulate(std::begin(features) + 1, std::end(features), 0UL, 
                    [](std::size_t s, std::vector<float> const& v){
                            return s + v.size();
                    });
    printf("total: %d\n", total);

    inputs.reserve(total);
	for (std::size_t i = 1; i < features.size(); i++) {
		inputs.insert(inputs.end(), features[i].begin(), features[i].end());
	}

    printf("input size: %d\n", inputs.size());
    // for (int i = 0; i < inputs.size(); i++){
    //     printf("%f ", inputs[i]);
    // }
    // printf("\n");

    return std::make_pair(inputs, label);
}

/*---------------- main function ---------------*/
int main(int argc, char **argv)
{
    // load/read the datafile
    std::ifstream file;
    std::string path_to_file = "./BostonHousing.csv";
    file.open(path_to_file, std::ios_base::in);

    // store data into a struct - pair
    std::pair<std::vector<float>, std::vector<float>> dataset = process_data(file);
}