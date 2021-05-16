#include <thallium.hpp>
#include <chrono>
#include <bits/stdc++.h>
#include "types.h"

namespace tl = thallium;

int main(int argc, char** argv) {
    // init the tl-client mode
    tl::engine myEngine("verbs", MARGO_CLIENT_MODE);
    tl::remote_procedure remote_do_rdma = myEngine.define("do_rdma").disable_response();
    tl::endpoint server_endpoint = myEngine.lookup(argv[1]);

    // we define a buffer with the content “Matthieu” (because it’s a string,
    // there is actually a null-terminating character). We then define
    // segments as a vector of pairs of void* and std::size_t
    std::string buffer = "Matthieu";
    std::vector<std::pair<void*, std::size_t>> segments(1);

    // Each segment (here only one) is characterized by its starting
    // address in local memory and its size. 
    segments[0].first  = (void*)(&buffer[0]);
    segments[0].second = buffer.size()+1;
    std::cout << "[CLIENT] num_characters = " << buffer.size()+1
              << ", size = " << sizeof(buffer)
              << std::endl;

    // We call engine::expose to expose the buffer and get a bulk instance from it.
    // We specify tl::bulk_mode::read_only to indicate that the memory will only be
    // read by other processes (alternatives are tl::bulk_mode::read_write
    // and tl::bulk_mode::write_only). 
    tl::bulk myBulk = myEngine.expose(segments, tl::bulk_mode::read_only);

    // Finally we send an RPC to the server, passing the bulk object as an argument.
    // Get back the arrival time at server
    remote_do_rdma.on(server_endpoint)(myBulk);

    return 0;
}