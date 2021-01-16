/*
 * Copyright (C) 2019  Hariharan Devarajan, Keith Bateman
 *
 * This file is part of HCL
 *
 * HCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <queue>
#include <fstream>
#include <mpi.h>

#include <hcl/common/data_structures.h>
#include <hcl/queue/queue.h>

struct ChamTaskType{
    // target entry point - function that holds the code that should be executed
    intptr_t tgt_entry_ptr;

    // we need index of image here as well since pointers are not matching for other ranks
    int32_t idx_image = 0;

    // task id (unique id that combines the host rank and a unique id per rank)
    int task_id;

    // number of arguments that should be passed to "function call" for target region
    int32_t arg_num;

    // host pointers will be used for transfer execution target region
    std::vector<void *> arg_hst_pointers;
    std::vector<int64_t> arg_sizes;
    std::vector<int64_t> arg_types;

    // target pointers will just be used at sender side for host pointer lookup 
    // and freeing of entries in data entry table
    std::vector<void *> arg_tgt_pointers;
    std::vector<ptrdiff_t> arg_tgt_offsets;

    int32_t is_remote_task      = 0;
    int32_t is_manual_task      = 0;
    int32_t is_replicated_task  = 0;
    int32_t is_migrated_task    = 0;
    int32_t is_cancelled        = 0;

    int32_t num_outstanding_recvbacks = 0;
    int32_t num_outstanding_replication_sends = 0;

    // Some special settings for stolen tasks
    int32_t source_mpi_rank             = 0;
    int32_t target_mpi_rank             = -1;
    int64_t buffer_size_output_data     = 0;

    // Mutex for either execution or receiving back/cancellation of a replicated task
    std::atomic<bool> result_in_progress;

    // Vector of replicating ranks
    std::vector<int> replication_ranks;

    size_t a;

    ChamTaskType():a(0) { }
    ChamTaskType(size_t a_):a(a_) { }

#ifdef HCL_ENABLE_RPCLIB
    MSGPACK_DEFINE(a);
#endif

    /* equal operator for comparing two Matrix. */
    bool operator==(const ChamTaskType &o) const {
        return a == o.a;
    }

    ChamTaskType& operator=( const ChamTaskType& other ) {
        a = other.a;
        return *this;
    }

    bool operator<(const ChamTaskType &o) const {
        return a < o.a;
    }

    bool operator>(const ChamTaskType &o) const {
        return a > o.a;
    }

    bool contains(const ChamTaskType &o) const {
        return a==o.a;
    }
};

namespace std {
    template<>
    struct hash<ChamTaskType> {
        size_t operator()(const ChamTaskType &k) const {
            return k.a;
        }
    };
}


int main (int argc,char* argv[])
{
    int provided;
    MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("Didn't receive appropriate MPI threading specification\n");
        exit(EXIT_FAILURE);
    }
    int comm_size, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int ranks_per_server = comm_size;
    int num_request = 10000;
    long size_of_request = 1000;
    bool debug = true;
    bool server_on_node = false;
    if(argc > 1)    ranks_per_server = atoi(argv[1]);
    if(argc > 2)    num_request = atoi(argv[2]);
    if(argc > 3)    size_of_request = (long)atol(argv[3]);
    if(argc > 4)    server_on_node = (bool)atoi(argv[4]);
    if(argc > 5)    debug = (bool)atoi(argv[5]);

    int len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &len);
    if (debug) {
        printf("%s/%d: %d\n", processor_name, my_rank, getpid());
    }

    if(debug && my_rank == 0){
        printf("%d ready for attach\n", comm_size);
        fflush(stdout);
        getchar();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    bool is_server = (my_rank+1) % ranks_per_server == 0;
    size_t my_server = my_rank / ranks_per_server;
    int num_servers = comm_size / ranks_per_server;

    // write server_list file
    if (is_server){
        std::ofstream server_list_file;
        server_list_file.open("./server_list");
        server_list_file << processor_name;
        server_list_file.close();
    }

    // The following is used to switch to 40g network on Ares.
    // This is necessary when we use RoCE on Ares.
    std::string proc_name = std::string(processor_name);
    /*int split_loc = proc_name.find('.');
    std::string node_name = proc_name.substr(0, split_loc);
    std::string extra_info = proc_name.substr(split_loc+1, string::npos);
    proc_name = node_name + "-40g." + extra_info;*/

    size_t size_of_elem = sizeof(int);

    printf("rank %d, is_server %d, my_server %zu, num_servers %d\n", my_rank, is_server, my_server, num_servers);

    const int array_size = TEST_REQUEST_SIZE;

    if (size_of_request != array_size) {
        printf("Please set TEST_REQUEST_SIZE in include/hcl/common/constants.h instead. Testing with %d\n", array_size);
    }

    std::array<int,array_size> my_vals=std::array<int,array_size>();

    HCL_CONF->IS_SERVER = is_server;
    HCL_CONF->MY_SERVER = my_server;
    HCL_CONF->NUM_SERVERS = num_servers;
    HCL_CONF->SERVER_ON_NODE = server_on_node || is_server;
    HCL_CONF->SERVER_LIST_PATH = "./server_list";

    // create hcl_queue with type is ChamTaskType
    hcl::queue<ChamTaskType> *queue;
    if (is_server) {
        queue = new hcl::queue<ChamTaskType>();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!is_server) {
        queue = new hcl::queue<ChamTaskType>();
    }

    // 
    std::queue<ChamTaskType> lqueue=std::queue<ChamTaskType>();

    MPI_Comm client_comm;
    MPI_Comm_split(MPI_COMM_WORLD, !is_server, my_rank, &client_comm);
    int client_comm_size;
    MPI_Comm_size(client_comm, &client_comm_size);
    MPI_Barrier(MPI_COMM_WORLD);

    if (!is_server) {
        Timer llocal_queue_timer=Timer();
        std::hash<KeyType> keyHash;
        /*Local std::queue test*/
        for(int i=0;i<num_request;i++){
            size_t val=my_server;
            llocal_queue_timer.resumeTime();
            size_t key_hash = keyHash(KeyType(val))%num_servers;
            if (key_hash == my_server && is_server){}
            lqueue.push(KeyType(val));
            llocal_queue_timer.pauseTime();
        }

        double llocal_queue_throughput=num_request/llocal_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        Timer llocal_get_queue_timer=Timer();
        for(int i=0;i<num_request;i++){
            size_t val=my_server;
            llocal_get_queue_timer.resumeTime();
            size_t key_hash = keyHash(KeyType(val))%num_servers;
            if (key_hash == my_server && is_server){}
            auto result = lqueue.front();
            lqueue.pop();
            llocal_get_queue_timer.pauseTime();
        }
        double llocal_get_queue_throughput=num_request/llocal_get_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        if (my_rank == 0) {
            printf("llocal_queue_throughput put: %f\n",llocal_queue_throughput);
            printf("llocal_queue_throughput get: %f\n",llocal_get_queue_throughput);
        }
        MPI_Barrier(client_comm);

        Timer local_queue_timer=Timer();
        uint16_t my_server_key = my_server % num_servers;
        /*Local queue test*/
        for(int i=0;i<num_request;i++){
            size_t val=my_server;
            auto key=KeyType(val);
            local_queue_timer.resumeTime();
            queue->Push(key, my_server_key);
            local_queue_timer.pauseTime();
        }
        double local_queue_throughput=num_request/local_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        Timer local_get_queue_timer=Timer();
        /*Local queue test*/
        for(int i=0;i<num_request;i++){
            size_t val=my_server;
            auto key=KeyType(val);
            local_get_queue_timer.resumeTime();
            size_t key_hash = keyHash(key)%num_servers;
            if (key_hash == my_server && is_server){}
            auto result = queue->Pop(my_server_key);
            local_get_queue_timer.pauseTime();
        }

        double local_get_queue_throughput=num_request/local_get_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        double local_put_tp_result, local_get_tp_result;
        if (client_comm_size > 1) {
            MPI_Reduce(&local_queue_throughput, &local_put_tp_result, 1,
                       MPI_DOUBLE, MPI_SUM, 0, client_comm);
            MPI_Reduce(&local_get_queue_throughput, &local_get_tp_result, 1,
                       MPI_DOUBLE, MPI_SUM, 0, client_comm);
            local_put_tp_result /= client_comm_size;
            local_get_tp_result /= client_comm_size;
        }
        else {
            local_put_tp_result = local_queue_throughput;
            local_get_tp_result = local_get_queue_throughput;
        }

        if (my_rank==0) {
            printf("local_queue_throughput put: %f\n", local_put_tp_result);
            printf("local_queue_throughput get: %f\n", local_get_tp_result);
        }

        MPI_Barrier(client_comm);

        Timer remote_queue_timer=Timer();
        /*Remote queue test*/
        uint16_t my_server_remote_key = (my_server + 1) % num_servers;
        for(int i=0;i<num_request;i++){
            size_t val = my_server+1;
            auto key=KeyType(val);
            remote_queue_timer.resumeTime();
            queue->Push(key, my_server_remote_key);
            remote_queue_timer.pauseTime();
        }
        double remote_queue_throughput=num_request/remote_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        MPI_Barrier(client_comm);

        Timer remote_get_queue_timer=Timer();
        /*Remote queue test*/
        for(int i=0;i<num_request;i++){
            size_t val = my_server+1;
            auto key=KeyType(val);
            remote_get_queue_timer.resumeTime();
            size_t key_hash = keyHash(key)%num_servers;
            if (key_hash == my_server && is_server){}
            queue->Pop(my_server_remote_key);
            remote_get_queue_timer.pauseTime();
        }
        double remote_get_queue_throughput=num_request/remote_get_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        double remote_put_tp_result, remote_get_tp_result;
        if (client_comm_size > 1) {
            MPI_Reduce(&remote_queue_throughput, &remote_put_tp_result, 1,
                       MPI_DOUBLE, MPI_SUM, 0, client_comm);
            remote_put_tp_result /= client_comm_size;
            MPI_Reduce(&remote_get_queue_throughput, &remote_get_tp_result, 1,
                       MPI_DOUBLE, MPI_SUM, 0, client_comm);
            remote_get_tp_result /= client_comm_size;
        }
        else {
            remote_put_tp_result = remote_queue_throughput;
            remote_get_tp_result = remote_get_queue_throughput;
        }

        if(my_rank == 0) {
            printf("remote queue throughput (put): %f\n",remote_put_tp_result);
            printf("remote queue throughput (get): %f\n",remote_get_tp_result);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    delete(queue);
    MPI_Finalize();
    exit(EXIT_SUCCESS);
}
