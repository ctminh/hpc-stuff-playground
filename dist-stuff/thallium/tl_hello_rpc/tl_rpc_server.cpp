#include <iostream>
#include <thallium.hpp>
#include <fstream>

namespace tl = thallium;

void hello(const tl::request& req) {
    std::cout << "Hello World!" << std::endl;
}

int main(int argc, char** argv) {

    // The engine::define method is used to define an RPC.
    // The first argument is the name of the RPC (a string),
    // the second is a function. This function should take a
    // const reference to a thallium::request as argument.
    tl::engine myEngine("tcp", THALLIUM_SERVER_MODE);
    myEngine.define("hello", hello).disable_response();

    // We will see in a future example what this request object is
    // used for. The disable_response() method is called to indicate
    // that the RPC is not going to send any response back to the client.

    std::cout << "Server running at address " << myEngine.self() << std::endl;

    // Try to write the server addr to file
    std::ofstream ser_addr_file;
    ser_addr_file.open ("./f_server_addr.txt");
    ser_addr_file << myEngine.self();
    ser_addr_file.close();

    return 0;
}