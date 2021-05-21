/*
 *
 * This example is referer from the queue_test example in part of hcl-src.
 *
 * HCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * https://github.com/scs-lab/hcl
 *
 */
#include <sys/types.h>
#include <iostream>
#include <string>
#include <cmath>
#include <sys/syscall.h>

// for mpi, omp
#include <mpi.h>
#include <omp.h>

// for hcl data-structures
#include <hcl/common/data_structures.h>
#include <hcl/queue/queue.h>

// for user-defined types
#include "types.h"

// for display hline
#define HLINE "-------------------------------------------------------------"

// ================================================================================
// Global Variables
// ================================================================================


// ================================================================================
// Main function
// ================================================================================
int main (int argc, char *argv[])
{
    /* /////////////////////////////////////////////////////////////////////////////
     * Initializing MPI
     * ////////////////////////////////////////////////////////////////////////// */

    // init mpi with mpi_init_thread
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("Didn't receive appropriate MPI threading specification\n");
        exit(EXIT_FAILURE);
    }

    // variables for tracking mpi-processes
    int comm_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    bool debug = true;

    // get hostname of each rank
    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);
    if (debug) {
        printf("[DBG] R%d: on %s | pid=%d\n", my_rank, processor_name, getpid());
    }

    // check num of ready ranks
    if(debug && my_rank == 0){
        printf("[DBG] %d ranks ready for HCL_CONF attaching\n", comm_size);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    /* /////////////////////////////////////////////////////////////////////////////
     * Configuring HCL-setup
     * ////////////////////////////////////////////////////////////////////////// */

    // assume that total ranks is even, 2 nodes for testing
    int ranks_per_server = comm_size / 2;

    // for simple each node has a server, so server_on_node is true
    bool server_on_node = true;

    // choose the server, for simple, assign R0 as the server,
    // for example, R0, R1 on node 1, R2, R3 on node 2,
    bool is_server = false;
    if (my_rank == 0 || my_rank == 2)
        is_server = true;

    // set my_server for R0, R1, others in the if
    int my_server = 0;
    if (my_rank > 1)
        my_server = 2;

    // ser num of servers for each rank
    int num_servers = 2;

    /* /////////////////////////////////////////////////////////////////////////////
     * Writing server addresses
     * ////////////////////////////////////////////////////////////////////////// */

    // write the server address into file
    if (is_server && my_rank == 0){
        std::ofstream server_list_file;
        server_list_file.open("./server_list");

        // temporarily put the hard-code ip of rome1, rome2 here
        // for tcp, we use processor_name and for simple, we use just R0 for writing
        server_list_file << "10.12.1.1" + "\n";
        server_list_file << "10.12.1.2";
        server_list_file.close();
    }
    std::cout << "[CHECK] R" << my_rank
              << ": is_server=" << is_server
              << ", my_server=" << my_server
              << ", num_servers=" << num_servers
              << ", server_on_node=" << server_on_node
              << std::endl;
    
    // just to make sure the shared-file system done in sync
    MPI_Barrier(MPI_COMM_WORLD);

    // configure hcl components before running, this configuration for each rank
    HCL_CONF->IS_SERVER = is_server;
    HCL_CONF->MY_SERVER = my_server;
    HCL_CONF->NUM_SERVERS = num_servers;
    HCL_CONF->SERVER_ON_NODE = server_on_node;
    HCL_CONF->SERVER_LIST_PATH = "./server_list";
    std::cout << HLINE << std::endl;

    /* /////////////////////////////////////////////////////////////////////////////
     * Creating HCL global queues over mpi ranks
     * ////////////////////////////////////////////////////////////////////////// */

    hcl::queue<Mattup_StdArr_t> *global_queue;

    // allocate the hcl queue at server-side
    if (is_server) {
        global_queue = new hcl::queue<Mattup_StdArr_t>();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // sounds like the server queue need to be created before the client ones
    if (!is_server) {
        global_queue = new hcl::queue<Mattup_StdArr_t>();
    }

    // declare a std-queue/rank at the local side for comparison
    std::queue<Mattup_StdArr_t> local_queue;

    /* /////////////////////////////////////////////////////////////////////////////
     * Split the mpi communicator from the server, just for client communicator
     * ////////////////////////////////////////////////////////////////////////// */
    MPI_Comm client_comm;
    MPI_Comm_split(MPI_COMM_WORLD, !is_server, my_rank, &client_comm);
    int client_comm_size;
    MPI_Comm_size(client_comm, &client_comm_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // check task size
    Mattup_StdArr_t tmp_T = Mattup_StdArr_t();
    size_t task_size = sizeof(tmp_T);
    std::cout << "[CHECK] task size = " << task_size << " bytes" << std::endl;
    std::cout << HLINE << std::endl;

    /* /////////////////////////////////////////////////////////////////////////////
     * Test throughput of the LOCAL QUEUES at client-side
     * ////////////////////////////////////////////////////////////////////////// */
    int num_tasks = 10;
    if (!is_server) {

        // for pushing local
        Timer t_push_local = Timer();
        for(int i = 0; i < num_tasks; i++){
            
            // allocate an arr_mat task
            Mattup_StdArr_t lT= Mattup_StdArr_t();
                       
            // put T into the queue and record eslapsed-time
            t_push_local.resumeTime();
            local_queue.push(lT);
            t_push_local.pauseTime();
        }
        double throughput_push_local = (num_tasks*task_size*1000) / (t_push_local.getElapsedTime()*1024*1024);
        std::cout << "[THROUGHPUT] R" << my_rank << ": local_push = " << throughput_push_local << " MB/s" << std::endl;

        // Barrier here for the client_commm
        MPI_Barrier(client_comm);

        // for deleting the local queue
        Timer t_pop_local = Timer();
        for (int i = 0;  i < num_tasks; i++){
            t_pop_local.resumeTime();
            auto result = local_queue.front();
            local_queue.pop();
            t_pop_local.pauseTime();
        }
        double throughput_pop_local = (num_tasks*task_size*1000) / (t_pop_local.getElapsedTime()*1024*1024);
        std::cout << "[THROUGHPUT] R" << my_rank << ": local_pop = " << throughput_pop_local << " MB/s" << std::endl;

        // Barrier here for the client_commm
        MPI_Barrier(client_comm);
        std::cout << HLINE << std::endl;
    }

    /* /////////////////////////////////////////////////////////////////////////////
     * Test throughput of the HCL QUEUES at client-side
     * /////////////////////////////////////////////////////////////////////////////  
     */
    if (!is_server) {
        
        // set a key by rank id
        uint16_t offset_key = my_server;

        // put tasks to the hcl-global-queue
        Timer t_push_remote = Timer();
        for(int i = 0; i < num_tasks; i++){

            // allocate the task
            Mattup_StdArr_t gT = Mattup_StdArr_t();
            
            // put tasks to the glob-queue and measure time
            t_push_remote.resumeTime();
            global_queue->Push(gT, offset_key);
            t_push_remote.pauseTime();
        }

        // estimate the remote-push throughput
        double throughput_push_remote = (num_tasks*task_size*1000) / (t_push_remote.getElapsedTime()*1024*1024);
        std::cout << "[THROUGHPUT] R" << my_rank << " [offset_key=" << offset_key << "]" << ": remote_push = "
                  << throughput_push_remote << " MB/s" << std::endl;
        
        // Barrier here for the client_commm
        MPI_Barrier(client_comm);
        
        // pop tasks from the hcl-global-queue
        Timer t_pop_remote = Timer();
        for(int i = 0; i < num_tasks; i++){
            
            // pop tasks and measure time
            t_pop_remote.resumeTime();
            auto loc_pop_res = global_queue->Pop(offset_key);
            t_pop_remote.pauseTime();
        }

        // estimate the remote-push throughput
        double throughput_pop_remote = (num_tasks*task_size*1000) / (t_pop_remote.getElapsedTime()*1024*1024);
        std::cout << "[THROUGHPUT] R" << my_rank << " [offset_key=" << offset_key << "]" << ": remote_pop = "
                  << throughput_pop_remote << " MB/s" << std::endl;
        
        // Barrier here for the client_commm
        MPI_Barrier(client_comm);
        std::cout << HLINE << std::endl;
    }

    // wait for make sure finalizing MPI safe
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    exit(EXIT_SUCCESS);
}
