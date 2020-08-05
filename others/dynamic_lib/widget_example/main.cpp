// main.cpp
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "./widget.hpp"

typedef void* dynamic_lib_handle;

// define some functions
static dynamic_lib_handle load_lib(const std::string& path);
static Widget* instantiate(const dynamic_lib_handle handle);

// define the struct
struct dynamic_lib {
    dynamic_lib_handle handle;
    std::string path;
    dynamic_lib(std::string p) : path(p), handle(nullptr) {}

    ~dynamic_lib() {
        if (handle != nullptr)
            close_lib(handle);
    }
};

int unique_signal = 42;

int main(int argc, char **argv)
{
    if (argc < 2) return 1;

    std::vector<dynamic_lib> libs;

    try {
        std::cout << "Opening: " << argv[1] << std::endl;
        std::ifstream fs(argv[1]);
        std::string tmp;

        // read from the file.
		while(std::getline(fs, tmp))
			libs.push_back(dynamic_lib(tmp));
    }
}