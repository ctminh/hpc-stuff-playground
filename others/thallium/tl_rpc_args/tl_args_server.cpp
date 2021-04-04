#include <thallium.hpp>
#include <fstream>

namespace tl = thallium;

/**
 * Notice that our sum function now takes two integers in addition to the const
 * reference to a thallium::request. You can also see that this request object
 * is used to send a response back to the client. Because the server now sends
 * something back to the client, we do not call ignore_response()
 * when defining the RPC.
 */
void sum(const tl::request& req, int x, int y) {
    std::cout << "Computing " << x << "+" << y << std::endl;
    req.respond(x+y);
}

int main(int argc, char** argv) {

    tl::engine myEngine("tcp://127.0.0.1:1234", THALLIUM_SERVER_MODE);
    std::cout << "Server running at address " << myEngine.self() << std::endl;

    // Try to write the server addr to file
    std::ofstream ser_addr_file;
    ser_addr_file.open ("./f_server_addr.txt");
    ser_addr_file << myEngine.self();
    ser_addr_file.close();

    // Define an rpc function - named sum
    myEngine.define("sum", sum);

    return 0;
}