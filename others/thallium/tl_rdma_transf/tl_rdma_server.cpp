#include <iostream>
#include <thallium.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <fstream>
#include <chrono>
#include <bits/stdc++.h>

// test including mpi/omp
#include <omp.h>
#include <mpi.h>

namespace tl = thallium;

int main(int argc, char** argv) {

    // Variables for tracking mpi-processes
    int my_rank;
    int num_ranks;
    int provided;   // level of provided thread support
    int requested = MPI_THREAD_MULTIPLE;    // level of desired thread support

    // Init MPI at runtime
    MPI_Init_thread(&argc, &argv, requested, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Init thallium engine
    tl::engine myEngine("tcp", THALLIUM_SERVER_MODE);

    // Try to write the server addr to file
    std::ofstream ser_addr_file;
    ser_addr_file.open ("./f_server_addr.txt");
    ser_addr_file << myEngine.self();
    ser_addr_file.close();

    std::function<void(const tl::request&, tl::bulk&)> f =
        [&myEngine](const tl::request& req, tl::bulk& b) {
            // get the client’s endpoint (client-addr)
            tl::endpoint ep = req.get_endpoint();

            // create a buffer of size 6. We initialize segments
            // and expose the buffer to get a bulk object from it.
            std::vector<char> v(6);
            std::vector<std::pair<void*, std::size_t>> segments(1);
            segments[0].first  = (void*)(&v[0]);
            segments[0].second = v.size();
            tl::bulk local = myEngine.expose(segments, tl::bulk_mode::write_only);

            // The call to the >> operator pulls data from the remote
            // bulk object b and the local bulk object. 
            b.on(ep) >> local;

            // Record end time at the server side
            clock_t end = clock();
            double db_end = double(end);
            // req.respond(db_end);
            std::cout << "[SERVER] end_time: " << db_end << std::endl;

            std::cout << "[SERVER] received bulk: ";
            for(auto c : v) std::cout << c;
            std::cout << std::endl;

            // Since the local bulk is smaller (6 bytes) than the remote
            // one (9 bytes), only 6 bytes are pulled. Hence the loop will
            // print Matthi. It is worth noting that an endpoint is needed
            // for Thallium to know in which process to find the memory
            // we are pulling. That’s what bulk::on(endpoint) does.
        };
    
    /** Understanding local and remote bulk objects
     * 
     * A bulk object created using engine::expose is local. When such a bulk
     * object is sent to another process, it becomes remote. Operations can only
     * be done between a local bulk object and a remote bulk object resolved
     * with an endpoint, e.g.,
     *      myRemoteBulk.on(myRemoteProcess) >> myLocalBulk;
     * or
     *      myLocalBulk >> myRemoteBulk.on(myRemoteProcess);
     * 
     */

    /** Transferring subsections of bulk objects
     * 
     * It is possible to select part of a bulk object to be transferred. This is
     * done as follows, for example.
     *      myRemoteBulk(3,45).on(myRemoteProcess) >> myLocalBulk(13,45);
     * 
     * Here we are pulling 45 bytes of data from the remote bulk starting at
     * offset 3 into the local bulk starting at its offset 13. We have
     * specified 45 as the number of bytes to be transferred. If the sizes
     * had been different, the smallest one would have been picked.
     */

    myEngine.define("do_rdma",f).disable_response();
}