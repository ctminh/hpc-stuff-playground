#include <thallium.hpp>

namespace tl = thallium;

int main(int argc, char** argv) {

    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <address>" << std::endl;
        exit(0);
    }

    // The client does not declare the hello function, since its code is on
    // the server. Instead, it calls engine::define with only the name of
    // the RPC, indicating that there exists on the server a RPC that goes
    // by this name. Again we call disable_response() to indicate that this RPC
    // does not send a response back. We then use the engine to perform an address
    // lookup. This call returns an endpoint representing the server.
    tl::engine myEngine("tcp", THALLIUM_CLIENT_MODE);
    tl::remote_procedure hello = myEngine.define("hello").disable_response();
    tl::endpoint server = myEngine.lookup(argv[1]);

    // Finally we can call the hello RPC by associating it with an endpoint.
    // hello.on(server) actually returns an instance of a class
    // callable_remote_procedure which has its parenthesis operator
    // overloaded to make it usable like a function.
    hello.on(server)();

    return 0;
}