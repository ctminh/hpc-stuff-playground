#include <iostream>
#include <thallium.hpp>
#include <fstream>

namespace tl = thallium;

int main(int argc, char** argv) {

    // Init thallium engine
    tl::engine myEngine("tcp", THALLIUM_SERVER_MODE);

    // Try to write the server addr to file
    std::ofstream ser_addr_file;
    ser_addr_file.open ("./f_server_addr.txt");
    ser_addr_file << myEngine.self();
    ser_addr_file.close();

    // The big advantage of lambdas is their ability to capture
    // local variables, which prevents the use of global variables
    // to pass user-provided data into RPC handlers.
    std::function<void(const tl::request&, int, int)> sum =
        [](const tl::request& req, int x, int y) {
            std::cout << "Computing " << x << "+" << y << std::endl;
            req.respond(x+y);
        };

    myEngine.define("sum", sum);

    return 0;
}