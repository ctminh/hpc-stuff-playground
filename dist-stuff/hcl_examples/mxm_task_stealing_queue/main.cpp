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

// for mpi, omp dependencies
#include <mpi.h>
#include <omp.h>

// for hcl data-structures
#include <hcl/common/data_structures.h>
#include <hcl/queue/queue.h>

#ifndef PARALLEL_OMP
#define PARALLEL_OMP 0
#endif

#ifndef TRACE
#define TRACE 0
#endif

#if TRACE==1

#include "VT.h"
static int _tracing_enabled = 1;

#ifndef VT_BEGIN_CONSTRAINED
#define VT_BEGIN_CONSTRAINED(event_id) if (_tracing_enabled) VT_begin(event_id);
#endif

#ifndef VT_END_W_CONSTRAINED
#define VT_END_W_CONSTRAINED(event_id) if (_tracing_enabled) VT_end(event_id);
#endif

#endif

// for user-defined types
#include "mxm_task_types.h"
#include "mxm_kernel.h"

// for other util-fnctions
#include "util.h"

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
        printf("[ERR] Didn't receive appropriate MPI threading specification\n");
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
    if(my_rank == 0){
        printf("[Usage] %d ranks ready for HCL_CONF attaching\n", comm_size);
        printf("[Usage] ./main <ranks_per_server> <server_on_node> <num_tasks> <num_threads> <server_list/hostfile>\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* /////////////////////////////////////////////////////////////////////////////
     * Configuring HCL-setup
     * ////////////////////////////////////////////////////////////////////////// */

    // get config-values from keyboard
    int num_tasks = 1000;
    int size_of_request = 1000;
    int ranks_per_node = 2;
    int num_nodes = comm_size / ranks_per_node;
    int ranks_per_server = ranks_per_node;
    bool server_on_node = false;
    int num_omp_threads = 1;
    std::string server_lists = "./server_list";
    
    if (argc > 1) ranks_per_server = atoi(argv[1]);
    if (argc > 2) server_on_node = (bool) atoi(argv[2]);
    if (argc > 3) num_tasks = atoi(argv[3]);
    if (argc > 4) num_omp_threads = atoi(argv[4]);
    if (argc > 5) server_lists = argv[5];

    // choose the server, for simple, assign R0 as the server,
    // for example, the last rank is the server
    bool is_server = (my_rank + 1) % ranks_per_server == 0;

    // set my_server for R0, R1, others in the if
    int my_server = my_rank / ranks_per_server;

    // ser num of servers for each rank
    int num_servers = comm_size / ranks_per_server;

    // get IB IP addresses
    MPI_Comm server_comm;
    MPI_Comm_split(MPI_COMM_WORLD, is_server, my_rank, &server_comm);
    int server_comm_size;
    MPI_Comm_size(server_comm, &server_comm_size);
    MPI_Barrier(MPI_COMM_WORLD);
    if (is_server){
        char *IPbuffer;
        IPbuffer = getHostIB_IPAddr();
        std::string send_addr(IPbuffer);
        std::cout << "[DBG] R" << my_rank << ": send_addr=" << send_addr << std::endl;
        size_t message_length = send_addr.size();
        char recv_buff[message_length*server_comm_size];
        MPI_Allgather(send_addr.c_str(), message_length, MPI_CHAR, recv_buff, message_length, MPI_CHAR, server_comm);
        if (my_rank == 1){
            // write ib-addresses to file
            ofstream ser_addr_file;
            ser_addr_file.open("./server_list");
            for (int i = 0;  i < num_servers; i++){
                std::string ib_addr = "";
                for (int j = 0; j < message_length; j++)
                    ib_addr = ib_addr + recv_buff[i*message_length + j];
                std::cout << "[DBG] Server " << i << ": IB-IP=" << ib_addr << std::endl;
                ser_addr_file << ib_addr << std::endl;
            }
            ser_addr_file.close();
        }
        MPI_Barrier(server_comm);

        /* try to write hostname on coolmuc
        std::string send_hostname(processor_name);
        std::cout << "[DBG] R" << my_rank << ": send_hostname=" << send_hostname << std::endl;
        size_t hn_length = send_hostname.size();
        char recv_buff[hn_length * server_comm_size];
        MPI_Allgather(send_hostname.c_str(), hn_length, MPI_CHAR, recv_buff, hn_length, MPI_CHAR, server_comm);
        if (my_rank == 1){
            // write hostname to file
            ofstream ser_addr_file;
            ser_addr_file.open("./server_list");
            for (int i = 0;  i < num_servers; i++){
                std::string hostname_addr = "";
                for (int j = 0; j < hn_length; j++)
                    hostname_addr = hostname_addr + recv_buff[i*hn_length + j];
                std::cout << "[DBG] Server " << i << ": hostname=" << hostname_addr << std::endl;
                ser_addr_file << hostname_addr << std::endl;
            }
            ser_addr_file.close();
        } */
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // configure hcl components before running, this configuration for each rank
    auto mem_size = SIZE * SIZE * (comm_size + 1) * num_tasks;
    HCL_CONF->IS_SERVER = is_server;
    HCL_CONF->MY_SERVER = my_server;
    HCL_CONF->NUM_SERVERS = num_servers;
    HCL_CONF->SERVER_ON_NODE = server_on_node;
    HCL_CONF->SERVER_LIST_PATH = "./server_list";
    HCL_CONF->MEMORY_ALLOCATED = mem_size;

    if (my_rank == 0){
        printf("[HCL_CONF] Rank | is_server | my_server | num_servers | server_on_node | mem_alloc\n");
    }
    printf("R%d %d %d %d %d %lu\n", my_rank, HCL_CONF->IS_SERVER, HCL_CONF->MY_SERVER, HCL_CONF->NUM_SERVERS,
                                    HCL_CONF->SERVER_ON_NODE, HCL_CONF->MEMORY_ALLOCATED);
    std::cout << HLINE << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    /* /////////////////////////////////////////////////////////////////////////////
     * Creating HCL global queues over mpi ranks
     * ////////////////////////////////////////////////////////////////////////// */

    hcl::queue<MatTask_Type> *global_queue;

    // allocate the hcl queue at server-side
    if (is_server) {
        global_queue = new hcl::queue<MatTask_Type>();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // sounds like the server queue need to be created before the client ones
    if (!is_server) {
        global_queue = new hcl::queue<MatTask_Type>();
    }

    /* /////////////////////////////////////////////////////////////////////////////
     * Split the mpi communicator from the server, just for client communicator
     * ////////////////////////////////////////////////////////////////////////// */

    MPI_Comm client_comm;
    MPI_Comm_split(MPI_COMM_WORLD, !is_server, my_rank, &client_comm);
    int client_comm_size;
    MPI_Comm_size(client_comm, &client_comm_size);
    MPI_Barrier(MPI_COMM_WORLD);

    /* /////////////////////////////////////////////////////////////////////////////
     * Main loop for creating tasks on each compute rank/ client rank
     * ////////////////////////////////////////////////////////////////////////// */

    if (!is_server) { /* IF NOT THE SERVER */

        // Set num of omp threads here, default is 1
        const int NTHREADS = num_omp_threads;

        // use the local key to push tasks on each server side
        std::cout << "[PUSH] R" << my_rank << ", NUM_OMP_THREADS=" << NTHREADS
                  << ": is creating " << num_tasks << " mxm-tasks..." << std::endl;
        uint16_t my_server_key = my_server % num_servers;

#if PARALLEL_OMP==1
    #pragma omp parallel num_threads(NTHREADS)
    {
        #pragma omp for
#endif
        for (int i = 0; i < num_tasks; i++){

            // int thread_id = omp_get_thread_num();
            // std::cout << "[PUSH] R" << my_rank
            //     << "-Thread " << thread_id << ": is pushing Task " << i
            //     << " into the global-queue..." << std::endl;

            // init the tasks with their values = their rank idx
            size_t val = my_rank;
            auto key = MatTask_Type(i, val);

            // push task to the global queue of each server
            global_queue->Push(key, my_server_key);
        }
#if PARALLEL_OMP==1
    }
#endif

        MPI_Barrier(client_comm);

        // pop tasks from the queue and then execute them
        std::cout << "[POP] R" << my_rank << ", NUM_OMP_THREADS=" << NTHREADS
                  << ": is getting " << num_tasks << " mxm-tasks out for executing..." << std::endl;

#if PARALLEL_OMP==1
    #pragma omp parallel num_threads(NTHREADS)
    {
        #pragma omp for
#endif
        for (int i = 0; i < num_tasks; i++) {

            // int thread_id = omp_get_thread_num();
            // std::cout << "[PUSH] R" << my_rank
            //     << "-Thread " << thread_id << ": is popping Task " << i
            //     << " out of the global-queue..." << std::endl;

            MatTask_Type tmp_pop_T;
            auto pop_result = global_queue->Pop(my_server_key);
            tmp_pop_T = pop_result.second;
        }
#if PARALLEL_OMP==1
    }
#endif

    } /* ENDIF NOT THE SERVER */

    // wait for making sure finalizing MPI safe
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    exit(EXIT_SUCCESS);
}
