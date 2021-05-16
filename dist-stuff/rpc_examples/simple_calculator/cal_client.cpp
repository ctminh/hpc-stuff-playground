#include <iostream>
#include "rpc/client.h"
#include "rpc/rpc_error.h"

int main(int argc, char *argv[])
{
    rpc::client c("localhost", rpc::constants::DEFAULT_PORT);

    try {
        std::cout << "add(2.2,3.3) = ";
        double add_res = c.call("add", 2.2, 3.3).as<double>();
        std::cout << add_res << std::endl;

        std::cout << "sub(3.3,2.2) = ";
        double sub_res = c.call("sub", 3.3, 2.2).as<double>();
        std::cout << sub_res << std::endl;

        std::cout << "mul(5.5,2.2) = ";
        double mul_res = c.call("mul", 5.5, 2.2).as<double>();
        std::cout << mul_res << std::endl;

        std::cout << "div(8.8,2.2) = ";
        double div_res = c.call("div", 8.8, 2.2).as<double>();
        std::cout << div_res << std::endl;

    } catch (rpc::rpc_error &e) {
        std::cout << std::endl << e.what() << std::endl;
        std::cout << "in function '" << e.get_function_name() << "': ";

        using err_t = std::tuple<int, std::string>;
        auto err = e.get_error().as<err_t>();
        std::cout << "[error " << std::get<0>(err) << "]: " << std::get<1>(err) << std::endl;
        return 1;
    }

    return 0;
}