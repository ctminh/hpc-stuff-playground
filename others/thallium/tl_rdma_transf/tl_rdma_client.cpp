#include <iostream>
#include <thallium.hpp>
#include <chrono>
#include <bits/stdc++.h>

namespace tl = thallium;

int main(int argc, char** argv) {
    // init the tl-client mode
    tl::engine myEngine("tcp", MARGO_CLIENT_MODE);
    tl::remote_procedure remote_do_rdma = myEngine.define("do_rdma");
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
    std::cout << "[CLIENT] data_size = " << buffer.size()+1 << std::endl;

    // We call engine::expose to expose the buffer and get a bulk instance from it.
    // We specify tl::bulk_mode::read_only to indicate that the memory will only be
    // read by other processes (alternatives are tl::bulk_mode::read_write
    // and tl::bulk_mode::write_only). 
    tl::bulk myBulk = myEngine.expose(segments, tl::bulk_mode::read_only);

    // Record start time before sending data
    clock_t start = clock();
    double db_start, db_end;
    db_start = double(start);

    // Finally we send an RPC to the server, passing the bulk object as an argument.
    // Get back the arrival time at server
    db_end = remote_do_rdma.on(server_endpoint)(myBulk);

    // Calculate the elapsed time
    double elapsed_time = (db_end - db_start) / double(CLOCKS_PER_SEC);
    std::cout << "[CLIENT] Elapsed-transf-time = " << elapsed_time  << " sec" << std::endl;

    return 0;
}