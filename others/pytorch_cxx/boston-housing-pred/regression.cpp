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

            m_data.clear()

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

std::istream &operator >> (std::istream& str, CSVRow& data)
{
	data.readNextRow(str);
	return str;
}

/*---------------- process data ----------------*/
std::pair<std::vector<float>, std::vector<float>> process_data(std::ifstream &file){
    std::vector<std::vector<float>> features;
    std::vector<float> label;

    CSVRow row;
    file >> row;    // ignore the first row
    while (file >> row) {
		features.emplace_back();
		for (std::size_t loop = 0;loop < row.size(); ++loop) {
			features.back().emplace_back(row[loop]);
		}
		features.back() = normalize_feature(features.back());
		
		// Push final column to label vector
		label.push_back(row[row.size()-1]);
	}

    for (int i = 0; i < label.size(); i++){
        cout << label.at(i) << std::endl;
    }

}

/*---------------- main function ---------------*/
int main(int argc, char **argv)
{
    // load/read the datafile
    std::ifstream file;
    std::string path_to_file = "./BostonHousing.csv"
    file.open(path_to_file, std::ios_base::in);

    // store data into a struct - pair
    std::pair<std::vector<float>, std::vector<float>> dataset = process_data(file);
}