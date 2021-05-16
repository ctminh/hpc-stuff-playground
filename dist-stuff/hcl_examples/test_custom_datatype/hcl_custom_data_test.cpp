/**
 * This example is to try some custom types with hcl queue
 * 
 * Different user-defined types are declared in types.h, we could 
 * add more struct/class-defined there.
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
    /* /////////////////////////////////////////////////////////////
     * Init and setup MPI
     * /////////////////////////////////////////////////////////////
     */
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

    // get hostname of each rank, on beast, hcl works with the direct ib-ip-addresses
    bool debug = true;
    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);
    if (debug) {
        printf("[DBG] R%d: on %s | pid=%d\n", my_rank, processor_name, getpid());
    }

    /* /////////////////////////////////////////////////////////////
     * Init and configs HCL-stuff
     * /////////////////////////////////////////////////////////////
     */
    // variables for configuring the hcl-modules
    char node1[10] = "10.12.1.1";
    char node2[10] = "10.12.1.2";
    int num_nodes = 1;
    if (argc == 2)  // try on rome1, rome2 of beast-system
        num_nodes = atoi(argv[1]);
    int ranks_per_server = comm_size / num_nodes;   // default is total-num of ranks
    bool server_on_node = false;

    // check num of ready ranks when running this program
    if(debug && my_rank == 0){
        printf("[DBG] %d ranks ready for HCL_CONF attaching\n", comm_size);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // choose the server, for simple, assign R0 as the server, or make the last as the server
    // by e.g., ((my_rank+1) % ranks_per_server) == 0;
    bool is_server = false;
    if (my_rank == 0)
        is_server = true;
    int num_servers = 1;
    int my_server = 0;

    // write the server address into file
    if (is_server && my_rank < ranks_per_server){
        std::ofstream server_list_file;
        server_list_file.open("./server_list");
        // temporarily put the hard-code ip of rome1 here
        server_list_file << node1; // processor_name;
        server_list_file.close();
    }
    std::cout << "[CHECK] R" << my_rank << ": is_server=" << is_server
              << ", my_server=" << my_server
              << ", num_servers=" << num_servers
              << std::endl;
    
    // just to make sure the shared-file system done in sync
    MPI_Barrier(MPI_COMM_WORLD);

    // configure hcl components before running, this configuration for each rank
    HCL_CONF->IS_SERVER = is_server;
    HCL_CONF->MY_SERVER = my_server;
    HCL_CONF->NUM_SERVERS = num_servers;
    // for the first 2 ranks, because R0 is server, R0&R1 are on Node 1
    if (my_rank < ranks_per_server)
        HCL_CONF->SERVER_ON_NODE = true;
    else
        HCL_CONF->SERVER_ON_NODE = false;
    HCL_CONF->SERVER_LIST_PATH = "./server_list";
    std::cout << HLINE << std::endl;


    /* /////////////////////////////////////////////////////////////
     * Create local-queue and hcl-global-queues
     * /////////////////////////////////////////////////////////////
     */
    hcl::queue<Single_DB_T> *global_queue;
    if (is_server) {    // at the server-side
        global_queue = new hcl::queue<Single_DB_T>();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (!is_server) {   // at the client-side
        global_queue = new hcl::queue<Single_DB_T>();
    }

    // declare a std-queue/rank at the local side for comparison
    std::queue<Single_DB_T> local_queue;

    // split the mpi communicator from the server, here is just for client communicator
    MPI_Comm client_comm;
    MPI_Comm_split(MPI_COMM_WORLD, !is_server, my_rank, &client_comm);
    int client_comm_size;
    MPI_Comm_size(client_comm, &client_comm_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // check task size
    Single_DB_T tmp_T = Single_DB_T();
    size_t task_size = sizeof(tmp_T);
    std::cout << "[CHECK] task size = " << task_size << " bytes" << std::endl;
    std::cout << HLINE << std::endl;

    /* /////////////////////////////////////////////////////////////
     * Test throughput of the LOCAL QUEUES at client-side
     * /////////////////////////////////////////////////////////////
     */
    int num_tasks = 10;
    if (!is_server) {
        // for pushing local
        Timer t_push_local = Timer();
        for(int i = 0; i < num_tasks; i++){
            
            // allocate an arr_mat task
            Single_DB_T lT= Single_DB_T();
                       
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

    /* /////////////////////////////////////////////////////////////
     * Test throughput of the HCL QUEUES at client-side
     * /////////////////////////////////////////////////////////////
     */
    if (!is_server) {
        
        // set a key by rank id
        uint16_t offset_key = my_rank;

        // put tasks to the hcl-global-queue
        Timer t_push_remote = Timer();
        for(int i = 0; i < num_tasks; i++){
            // allocate the task
            Single_DB_T gT = Single_DB_T();

            printf("[DBG] R%d pushes task-%d into the global-hcl queue...\n", my_rank, i);
            
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
