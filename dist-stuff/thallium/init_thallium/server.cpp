#include <iostream>
#include <thallium.hpp>
#include <fstream>

/**
 * Thallium’s main class is the engine. It is used to initialize the underlying
 * libraries (Margo, Mercury, and Argobots), to define RPCs, expose segments
 * of memory for RDMA, and lookup addresses.
 */
namespace tl = thallium;

int main(int argc, char** argv) {

    // The first argument of the constructor is the server’s protocol (tcp).
    // You can also provide a full address (e.g. tcp://127.0.0.1:1234)
    // in particular if you want to force using a particular port number.
    // You can refer to the Mercury documentation to see a list of available
    // protocols. The second argument, THALLIUM_SERVER_MODE, indicates that
    // this engine is a server. When running this program, it will print
    // the server’s address, then block on the destructor of myEngine.
    // Server engines are indeed supposed to wait for incoming RPCs.
    // We will see in the next tutorial how to properly shut it down.

    tl::engine myEngine("tcp", THALLIUM_SERVER_MODE);
    std::cout << "[SERVER] running at address " << myEngine.self() << std::endl;

    // Try to write the server addr to file
    std::ofstream ser_addr_file;
    ser_addr_file.open ("./f_server_addr.txt");
    ser_addr_file << myEngine.self();
    ser_addr_file.close();

    return 0;
}