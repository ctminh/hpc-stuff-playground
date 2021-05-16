#include <thallium.hpp>

/**
 * The following code initialize the engine as a client.
 */
namespace tl = thallium;

int main(int argc, char** argv) {

    // Contrary to the server, this program will exit normally.
    // Client engine are not supposed to wait for anything. We use
    // THALLIUM_CLIENT_MODE to specify that the engine is a client.
    // We can simply provide the protocol, here “tcp”, since a
    // client is not receiving on any address.
    tl::engine myEngine("tcp", THALLIUM_CLIENT_MODE);

    return 0;
}
