#include <cstdio>
#include <cstdlib>
#include <random>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <sys/syscall.h>
#include <atomic>
#include <chrono>

// for mpi, omp
#include <omp.h>
#include <mpi.h>

// for thallium
#include <thallium.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <bits/stdc++.h>

// for boost-serialize data
#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

// declare some namespaces
namespace tl = thallium;

// ================================================================================
// Main function
// ================================================================================
int main(int argc, char **argv){

    // variables for tracking mpi-processes
    int my_rank;
    int num_ranks;
    int provided;   // level of provided thread support
    int requested = MPI_THREAD_MULTIPLE;    // level of desired thread support

    // init MPI at runtime
    MPI_Init_thread(&argc, &argv, requested, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    /*
     * **************************************************************************** 
     * Init thallium rpc-rdma data transfer
     * ****************************************************************************
     */
    
    // if rank = 0, init thallium server
    if (my_rank == 0){
        // check the server
        int name_len;
        char hostname[MPI_MAX_PROCESSOR_NAME];
        MPI_Get_processor_name(hostname, &name_len);
        std::string server_addr_str(hostname);
        std::cout << "[R0] is initializing the tl-server at " << server_addr_str << std::endl;

        // init the tl-server mode
        tl::engine ser_engine("mpi+static", THALLIUM_SERVER_MODE);
        std::cout << "[R0] inits tl-server at " << ser_engine.self() << std::endl;
        std::string str_serveraddr = ser_engine.self();
        std::cout << "[R0] casts the addr to string-type: " << str_serveraddr << std::endl;

        // define rpc-function at the server side
        std::function<void(const tl::request&, tl::bulk&)> f =
        [&ser_engine](const tl::request& req, tl::bulk& b) {
            // get the client’s endpoint (client-addr)
            tl::endpoint ep = req.get_endpoint();

            // create a buffer of size 6. We initialize segments
            // and expose the buffer to get a bulk object from it.
            std::vector<char> v(6);
            std::vector<std::pair<void*, std::size_t>> segments(1);
            segments[0].first  = (void*)(&v[0]);
            segments[0].second = v.size();
            tl::bulk local = ser_engine.expose(segments, tl::bulk_mode::write_only);

            // The call to the >> operator pulls data from the remote
            // bulk object b and the local bulk object. 
            b.on(ep) >> local;

            std::cout << "[R0] SERVER received bulk: ";
            for(auto c : v) std::cout << c;
            std::cout << std::endl;

            // Since the local bulk is smaller (6 bytes) than the remote
            // one (9 bytes), only 6 bytes are pulled. Hence the loop will
            // print Matthi. It is worth noting that an endpoint is needed
            // for Thallium to know in which process to find the memory
            // we are pulling. That’s what bulk::on(endpoint) does.

            // finalize the server after the first transfer
            ser_engine.finalize();
        };

        // define the procedure
        ser_engine.define("do_rdma",f).disable_response();

        // use mpi_send to let the client know the server address
        int reciever = 1; // rank 1, send_tag = 0
        MPI_Send(str_serveraddr.c_str(), str_serveraddr.length(), MPI_CHAR, reciever, 0, MPI_COMM_WORLD);


    } else if (my_rank == 1) {
        // check the client
        std::cout << "[R1] is initializing the tl-client..." << std::endl;

        // use mpi probe to check the message-size from rank 0
        MPI_Status status;
        int sender = 0; // rank 0, send_tag = 0
        MPI_Probe(sender, 0, MPI_COMM_WORLD, &status);
        int mess_size;
        MPI_Get_count(&status, MPI_CHAR, &mess_size);

        // get the ser-addr over mpi-transfer
        char *rec_buf = new char[mess_size];
        MPI_Recv(rec_buf, mess_size, MPI_CHAR, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::string ser_addr(rec_buf, mess_size);
        std::cout << "[R1] got the serv-addr: " << ser_addr << std::endl;

        // init the tl-client mode
        tl::engine cli_engine("mpi+static", MARGO_CLIENT_MODE);
        tl::remote_procedure remote_do_rdma = cli_engine.define("do_rdma").disable_response();
        tl::endpoint ser_endpoint = cli_engine.lookup(ser_addr);

        // we define a buffer with the content “Matthieu” (because it’s a string,
        // there is actually a null-terminating character). We then define
        // segments as a vector of pairs of void* and std::size_t
        std::string buffer = "Matthieu";
        std::vector<std::pair<void*, std::size_t>> segments(1);

        // Each segment (here only one) is characterized by its starting
        // address in local memory and its size. 
        segments[0].first  = (void*)(&buffer[0]);
        segments[0].second = buffer.size()+1;
        std::cout << "[R1] CLIENT num_characters = " << buffer.size()+1
                << ", size = " << sizeof(buffer)
                << std::endl;

        // We call engine::expose to expose the buffer and get a bulk instance from it.
        // We specify tl::bulk_mode::read_only to indicate that the memory will only be
        // read by other processes (alternatives are tl::bulk_mode::read_write
        // and tl::bulk_mode::write_only). 
        tl::bulk myBulk = cli_engine.expose(segments, tl::bulk_mode::read_only);

        // Finally we send an RPC to the server, passing the bulk object as an argument.
        // Get back the arrival time at server
        remote_do_rdma.on(ser_endpoint)(myBulk);

        // free the memory allocated by new
        delete[] rec_buf; // because having [size] after new
    }
    /*
     * ****************************************************************************
     */

    // put mpi barrier here to wait, just make sure every ranks finished
    MPI_Barrier(MPI_COMM_WORLD);

    // finalize mpi
    MPI_Finalize();

    return 0;
}