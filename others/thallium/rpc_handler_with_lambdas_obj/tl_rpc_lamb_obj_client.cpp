#include <iostream>
#include <thallium.hpp>

namespace tl = thallium;

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <address>" << std::endl;
        exit(0);
    }

    // The client calls the remote procedure with two integers and gets an
    // integer back. This way of passing parameters and returning a value
    // hides many implementation details that are handled with a lot of
    // template metaprogramming. Effectively, what happens is the following.
    // 
    // When passing the sum function to engine::define, the compiler deduces
    // from its signature that clients will send two integers. Thus it creates
    // the code necessary to deserialize two integers before calling the function.
    tl::engine myEngine("tcp", THALLIUM_CLIENT_MODE);
    tl::remote_procedure sum = myEngine.define("sum");
    tl::endpoint server = myEngine.lookup(argv[1]);

    // On the client side, calling sum.on(server)(33, 66) makes the compiler
    // realize that the client wants to serialize two integers and send them along
    // with the RPC. It therefore also generates the code for that. The same happens
    // when calling req.respond(...) in the server, the compiler generates
    // the code necessary to serialize whatever object has been passed.

    int ret = sum.on(server)(33, 66);
    std::cout << "Server answered " << ret << std::endl;

    // Back on the client side, sum.on(server)(42,63) does not actually return
    // an integer. It returns an instance of thallium::packed_response, which
    // can be cast into any type, here an integer. Asking the packed_response
    // to be cast into an integer also instructs the compiler to generate
    // the right deserialization code.

    /**
     * Note: A common miskate consists of changing the arguments accepted by
     * an RPC handler but forgetting to update the calls to that RPC on clients.
     * This can lead to data corruptions or crashes. Indeed, Thallium has no
     * way to check that the types passed by the client to the RPC call are
     * the ones expected by the server.
     */

    /**
     * Another common mistake is to use integers of different size on client and
     * server. For example sum.on(server)(33, 66); on the client side will serialize
     * two int values, because int is the default for integer litterals. If the
     * corresponding RPC handler on the server side had been void
     * sum(const tl::request& req, int64_t x, int64_t y), the call would have
     * led to data corruptions and potential crash. One way to ensure that the
     * right types are used is to explicitely cast the litterals:
     * sum.on(server)(static_cast<int64_t>(33), static_cast<int64_t>(66));.
     */

    /**
     * Timeout: It can sometime be useful for an operation to be given a certain
     * amount of time before timing out. This can be done using the
     * callable_remote_procedure::timed() function. This function behaves
     * like the operator() but takes a first parameter of type
     * std::chrono::duration representing an amount of time after which
     * the call will throw a thallium::timeout exception. For instance
     * in the above client code, int ret = sum.on(server)(33, 66); would become
     * int ret = sum.on(server).timed(std::chrono::milliseconds(5), 33 ,66);
     * to allow for a 5ms
     */

    return 0;
}