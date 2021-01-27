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
#include <atomic>

#include <hcl/common/data_structures.h>
#include <hcl/queue/queue.h>


#ifndef MPI_BLOCKING
#define MPI_BLOCKING 1
#endif


/* struct for tasks */
typedef struct mig_task_t {
    // task id, img_idx, args-num
    int id;
    int32_t idx_image = 0;
    int32_t arg_num;
    std::vector<void *> arg_hst_pointers;
    std::vector<int64_t> arg_sizes;

#ifdef HCL_ENABLE_RPCLIB
    MSGPACK_DEFINE(id);
#endif
    
    // constructor 1
    mig_task_t(){
        id = 0;
        idx_image = 0;
        arg_num = 0;
        void *ptr = nullptr;
        arg_hst_pointers.push_back(ptr);
        arg_sizes.push_back(0);
        arg_sizes.push_back(0);
        arg_sizes.push_back(0);
        arg_sizes.push_back(0);
        arg_sizes.push_back(0);
        arg_sizes.push_back(0);
    }

    // constructor 2
    mig_task_t(int32_t img, int task_id, int32_t num_args){
        id = task_id;
        idx_image = img;
        arg_num = num_args;
        void *ptr = nullptr;
        arg_hst_pointers.push_back(ptr);
        arg_sizes.push_back(10240);
        arg_sizes.push_back(20480);
        arg_sizes.push_back(40960);
        arg_sizes.push_back(51200);
        arg_sizes.push_back(102400);
        arg_sizes.push_back(204800);
    }

} mig_task_t;

/* basic struct for testing queue from hcl-src */
typedef struct KeyType{
    int a;

#ifdef HCL_ENABLE_RPCLIB
    MSGPACK_DEFINE(a);
#endif

    // constructor 1
    KeyType(){
        a = 0;
    }
    // constructor 2
    KeyType(int val){
        a = val;
    }

} KeyType;


/* encode data before sending */
void *encode_send_buffer(mig_task_t **tasks, int32_t num_tasks, int32_t *buffer_size){
    // calculate the size
    int total_size = sizeof(int32_t);   // 0. num of tasks
    for (int i = 0; i < num_tasks; i++){
        total_size += sizeof(int)       // 1. task_id   : int
                + sizeof(int32_t)       // 2. idx_image : int32_t
                + sizeof(int32_t)       // 3. arg_num   : int32_t
                + sizeof(void)          // 4. it's null-void ptr, so dont need to pack  : std::vector<void *>
                + tasks[i]->arg_num * sizeof(int64_t);   // 5. list of arg_sizes        : std::vector<int64_t>
    }

    // allocate mem for transfering data
    char *buff = (char *) malloc(total_size);
    char *cur_ptr = (char *) buff;

    // contain the value # tasks in this message
    ((int32_t *) cur_ptr)[0] = num_tasks;
    cur_ptr += sizeof(int32_t);

    for (int i = 0; i < num_tasks; i++){
        // 1. task id
        ((int *) cur_ptr)[0] = tasks[i]->id;
        cur_ptr += sizeof(int);

        // 2. idx image
        ((int32_t *) cur_ptr)[0] = tasks[i]->idx_image;
        cur_ptr += sizeof(int32_t);

        // 3. arg_num
        ((int32_t *) cur_ptr)[0] = tasks[i]->arg_num;
        cur_ptr += sizeof(int32_t);

        // 4. arg_hst_pointers
        memcpy(cur_ptr, &(tasks[i]->arg_hst_pointers[0]), sizeof(void));
        cur_ptr += sizeof(void);

        // 5. arg_sizes
        memcpy(cur_ptr, &(tasks[i]->arg_sizes[0]), tasks[i]->arg_num * sizeof(int64_t));
        cur_ptr += tasks[i]->arg_num * sizeof(int64_t);
    }

    *buffer_size = total_size;
    return buff;
}


/* decode data before sending */
void decode_send_buffer(void *buffer, int mpi_tag, int32_t *num_tasks, std::vector<mig_task_t *> &recv_tasks){
    // current pointer position
    char *cur_ptr = (char *) buffer;
    int num_recv_bytes = 0;

    // 0. num of recv_tasks
    int n_tasks = ((int32_t *) cur_ptr)[0];
    cur_ptr += sizeof(int32_t);
    num_recv_bytes += sizeof(int32_t);

    *num_tasks = n_tasks;

    // decode task by task
    for (int i = 0; i < n_tasks; i++){
        mig_task_t *task = new mig_task_t();

        // 1. task id
        task->id = ((int *) cur_ptr)[0];
        cur_ptr += sizeof(int);
        num_recv_bytes += sizeof(int);

        // 2. idx image
        task->idx_image = ((int32_t *) cur_ptr)[0];
        cur_ptr += sizeof(int32_t);
        num_recv_bytes += sizeof(int32_t);

        // 3. arg_num
        task->arg_num = ((int32_t *) cur_ptr)[0];
        cur_ptr += sizeof(int32_t);
        num_recv_bytes += sizeof(int32_t);

        // 4. arg_hst_pointers
        memcpy(&(task->arg_hst_pointers[0]), cur_ptr, sizeof(void));
        cur_ptr += sizeof(void);
        num_recv_bytes += sizeof(void);

        // 5. arg_sizes
        memcpy(&(task->arg_sizes[0]), cur_ptr, task->arg_num * sizeof(int64_t));
        cur_ptr += task->arg_num * sizeof(int64_t);
        num_recv_bytes += task->arg_num * sizeof(int64_t);

        recv_tasks.push_back(task);
    }
}


/* send tasks to a target rank */
int offload_tasks_to_rank(int my_rank, mig_task_t **tasks, int32_t num_tasks, int target_rank){
    
    int32_t buffer_size = 0;
    void *buffer = NULL;
    int tmp_tag = my_rank;

    // encode buffer before sending tasks
    buffer = encode_send_buffer(tasks, num_tasks, &buffer_size);

    // set n_requests
    int n_requests = num_tasks;
    MPI_Request *requests = new MPI_Request[n_requests];

#if MPI_BLOCKING
    MPI_Send(buffer, buffer_size, MPI_BYTE, target_rank, tmp_tag, MPI_COMM_WORLD);
#else
    MPI_Isend(buffer, buffer_size, MPI_BYTE, target_rank, tmp_tag, MPI_COMM_WORLD, &requests[0]);
#endif

    return 0;
}


/* recv tasks from a source-rank */
void recv_tasks_from_rank(void *buffer, int tag, int source, int recv_buff_size, std::vector<mig_task_t *> &list_recv_tasks){
    MPI_Request request = MPI_REQUEST_NULL;

#if MPI_BLOCKING
    int res = MPI_Recv(buffer, recv_buff_size, MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
    int res = MPI_Irecv(buffer, recv_buff_size, MPI_BYTE, source, tag, MPI_COMM_WORLD, &request);
#endif

    int num_tasks;
    decode_send_buffer(buffer, tag, &num_tasks, list_recv_tasks);

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
    bool debug = false;
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

    printf("rank %d, is_server %d, my_server %zu, num_servers %d\n", my_rank, is_server, my_server, num_servers);

    HCL_CONF->IS_SERVER = is_server;
    HCL_CONF->MY_SERVER = my_server;
    HCL_CONF->NUM_SERVERS = num_servers;
    HCL_CONF->SERVER_ON_NODE = server_on_node || is_server;
    HCL_CONF->SERVER_LIST_PATH = "./server_list";

    // create hcl_queue with type is ChamTaskType
    hcl::queue<mig_task_t> *glob_queue;
    // hcl::queue<KeyType> *dist_keytype_queue;
    if (is_server) {
        glob_queue = new hcl::queue<mig_task_t>();
        // dist_keytype_queue = new hcl::queue<KeyType>();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!is_server) {
        glob_queue = new hcl::queue<mig_task_t>();
        // dist_keytype_queue = new hcl::queue<KeyType>();
    }

    // create local queue per rank
    std::queue<mig_task_t> loca_queue = std::queue<mig_task_t>();

    // communicator for clients
    MPI_Comm client_comm;
    MPI_Comm_split(MPI_COMM_WORLD, !is_server, my_rank, &client_comm);
    int client_comm_size;
    MPI_Comm_size(client_comm, &client_comm_size);
    MPI_Barrier(MPI_COMM_WORLD);

    /* ///////////////// Test Throughput of Local Queue per Rank ///////////////// */
    if (!is_server) {
        Timer timer_loca_queue_put = Timer();
        size_t size_of_task = 0;
        for(int i = 0; i < num_request; i++){
            mig_task_t task = mig_task_t(1, i, 5);
            size_of_task = sizeof(task);
            // measure time of put
            timer_loca_queue_put.resumeTime();
            loca_queue.push(task);
            timer_loca_queue_put.pauseTime();
        }
        double throughput_loca_queue_put = (num_request*size_of_task*1000) / (timer_loca_queue_put.getElapsedTime()*1024*1024);

        Timer timer_loca_queue_get = Timer();
        for(int i = 0; i < num_request; i++){
            // measure time of get
            timer_loca_queue_get.resumeTime();
            auto result = loca_queue.front();
            loca_queue.pop();
            timer_loca_queue_get.pauseTime();
        }
        double throughput_loca_queue_get = (num_request*size_of_task*1000) / (timer_loca_queue_get.getElapsedTime()*1024*1024);

        // print the throughput-results
        if (my_rank == 0) {
            printf("throughput_loca_queue put: %f (MB/s)\n", throughput_loca_queue_put);
            printf("throughput_loca_queue get: %f (MB/s)\n",throughput_loca_queue_get);
        }

        /* ///////////////// Test Throughput of Global Queue per Rank ///////////////// */
        /* ///////////////// LOCAL ACCESS on Global Queue             ///////////////// */
        uint16_t my_server_key = my_server % num_servers;
        printf("[CHECK] local-access glob_queue: R%d -> glob_queue with key = %d | server=%ld\n", my_rank, my_server_key, my_server);

        Timer timer_glob_queue_loc_put = Timer();
        for(int i = 0; i < num_request; i++){
            mig_task_t task = mig_task_t(1, i, 5);
            timer_glob_queue_loc_put.resumeTime();
            glob_queue->Push(task, my_server_key);
            timer_glob_queue_loc_put.pauseTime();
        }
        double throughput_glob_queue_loc_put = (num_request*size_of_task*1000) / (timer_glob_queue_loc_put.getElapsedTime()*1024*1024);

        Timer timer_glob_queue_loc_get = Timer();
        for(int i = 0; i < num_request; i++){
            timer_glob_queue_loc_get.resumeTime();
            auto glob_pop_result = glob_queue->Pop(my_server_key);
            timer_glob_queue_loc_get.pauseTime();
        }
        double throughput_glob_queue_loc_get = (num_request*size_of_task*1000) / (timer_glob_queue_loc_get.getElapsedTime()*1024*1024);

        // printf("R%d: throughput_glob_queue local-access with put: %f (MB/s)\n", my_rank, throughput_glob_queue_loc_put);
        // printf("R%d: throughput_glob_queue local-access with get: %f (MB/s)\n", my_rank, throughput_glob_queue_loc_get);

        // accumulate the throughput-results
        double total_throughput_glob_queue_loc_put, total_throughput_glob_queue_loc_get;
        if (client_comm_size > 1) {
            MPI_Reduce(&throughput_glob_queue_loc_put, &total_throughput_glob_queue_loc_put, 1,
                       MPI_DOUBLE, MPI_SUM, 0, client_comm);
            MPI_Reduce(&throughput_glob_queue_loc_get, &total_throughput_glob_queue_loc_get, 1,
                       MPI_DOUBLE, MPI_SUM, 0, client_comm);
            total_throughput_glob_queue_loc_put /= client_comm_size;
            total_throughput_glob_queue_loc_get /= client_comm_size;
        }
        else {
            total_throughput_glob_queue_loc_put = throughput_glob_queue_loc_put;
            total_throughput_glob_queue_loc_get = throughput_glob_queue_loc_get;
        }

        // print the throughput-results
        if (my_rank == 0){
            printf("total throughput_glob_queue local-access with put: %f (MB/s)\n", total_throughput_glob_queue_loc_put);
            printf("total throughput_glob_queue local-access with get: %f (MB/s)\n", total_throughput_glob_queue_loc_get);
        }

        MPI_Barrier(client_comm);


        /* ///////////////// REMOTE ACCESS on Global Queue             ///////////////// */
        uint16_t my_server_remote_key = (my_server + 1) % num_servers;
        printf("[CHECK] remote-access glob_queue: R%d -> glob_queue with key = %d | server=%ld\n", my_rank, my_server_remote_key, my_server);

        Timer timer_glob_queue_remote_put = Timer();
        for(int i = 0; i < num_request; i++){
            mig_task_t task = mig_task_t(1, i, 5);
            // measure time
            timer_glob_queue_remote_put.resumeTime();
            glob_queue->Push(task, my_server_remote_key);
            timer_glob_queue_remote_put.pauseTime();
        }
        double throughput_glob_queue_rem_put = (num_request*size_of_task*1000) / (timer_glob_queue_remote_put.getElapsedTime()*1024*1024);

        MPI_Barrier(client_comm);

        Timer timer_glob_queue_remote_get = Timer();
        for(int i = 0; i < num_request; i++){
            // measure time
            timer_glob_queue_remote_get.resumeTime();
            auto glob_rem_pop_result = glob_queue->Pop(my_server_remote_key);
            timer_glob_queue_remote_get.pauseTime();
        }
        double throughput_glob_queue_rem_get = (num_request*size_of_task*1000) / (timer_glob_queue_remote_get.getElapsedTime()*1024*1024);
        
        // printf("R%d: throughput_glob_queue remote-access with put: %f (MB/s)\n", my_rank, throughput_glob_queue_rem_put);
        // printf("R%d: throughput_glob_queue remote-access with get: %f (MB/s)\n", my_rank, throughput_glob_queue_rem_get);

        double total_throughput_glob_queue_rem_put, total_throughput_glob_queue_rem_get;
        if (client_comm_size > 1) {
            MPI_Reduce(&throughput_glob_queue_rem_put, &total_throughput_glob_queue_rem_put, 1,
                       MPI_DOUBLE, MPI_SUM, 0, client_comm);
            MPI_Reduce(&throughput_glob_queue_rem_get, &total_throughput_glob_queue_rem_get, 1,
                       MPI_DOUBLE, MPI_SUM, 0, client_comm);
            total_throughput_glob_queue_rem_put /= client_comm_size;
            total_throughput_glob_queue_rem_get /= client_comm_size;
        }
        else {
            total_throughput_glob_queue_rem_put = throughput_glob_queue_rem_put;
            total_throughput_glob_queue_rem_get = throughput_glob_queue_rem_get;
        }

        // print the throughput-results
        if (my_rank == 0){
            printf("total throughput_glob_queue remote-access with put: %f (MB/s)\n", total_throughput_glob_queue_rem_put);
            printf("total throughput_glob_queue remote-access with get: %f (MB/s)\n", total_throughput_glob_queue_rem_get);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ///////////////// Test HCL-dis-queue with bas-type  //////////////////////// */
    /* ///////////////// local/remote access ////////////////////////////////////// */

    // size_t keytype_size = sizeof(int);

    // if (!is_server){
    //     /* ///////////////// LOCAL ACCESS on Global Queue  ///////////////// */
    //     uint16_t server_key = my_server % num_servers;

    //     Timer t_keytype_queue_loc_put = Timer();
    //     for(int i = 0; i < num_request; i++){
    //         KeyType k = KeyType(my_rank);
    //         t_keytype_queue_loc_put.resumeTime();
    //         dist_keytype_queue->Push(k, server_key);
    //         t_keytype_queue_loc_put.pauseTime();
    //     }
    //     double throughput_loc_put = (num_request*keytype_size*1000) / (t_keytype_queue_loc_put.getElapsedTime()*1024*1024);

    //     Timer t_keytype_queue_loc_get = Timer();
    //     for(int i = 0; i < num_request; i++){
    //         t_keytype_queue_loc_get.resumeTime();
    //         auto k_pop_result = dist_keytype_queue->Pop(server_key);
    //         t_keytype_queue_loc_get.pauseTime();
    //     }
    //     double throughput_loc_get = (num_request*keytype_size*1000) / (t_keytype_queue_loc_get.getElapsedTime()*1024*1024);

    //     // accumulate the throughput-results
    //     double total_throughput_loc_put, total_throughput_loc_get;
    //     if (client_comm_size > 1) {
    //         MPI_Reduce(&throughput_loc_put, &total_throughput_loc_put, 1,
    //                    MPI_DOUBLE, MPI_SUM, 0, client_comm);
    //         MPI_Reduce(&throughput_loc_get, &total_throughput_loc_get, 1,
    //                    MPI_DOUBLE, MPI_SUM, 0, client_comm);
    //         total_throughput_loc_put /= client_comm_size;
    //         total_throughput_loc_get /= client_comm_size;
    //     }
    //     else {
    //         total_throughput_loc_put = throughput_loc_put;
    //         total_throughput_loc_get = throughput_loc_get;
    //     }

    //     // print the throughput-results
    //     if (my_rank == 0){
    //         printf("total throughput KEYTYPE-queue local-access with put: %f (MB/s)\n", total_throughput_loc_put);
    //         printf("total throughput KEYTYPE-queue local-access with get: %f (MB/s)\n", total_throughput_loc_get);
    //     }

    //     MPI_Barrier(client_comm);
    // }
    
    // MPI_Barrier(MPI_COMM_WORLD);

    /* ///////////////// Test 2-Sided Communication  ////////////////////////////// */
    /* ///////////////// Paired_Process for Send/Recv Tasks /////////////////////// */
    /* R0 <- R1, R2 <- R3                                                           */
    int recv_buff_size = 110;   // given temporarily
    Timer timer_2side_comm = Timer();
    for (int i = 0; i < num_request; i++){
        timer_2side_comm.resumeTime();
        if (my_rank % 2 == 0) // recieve tasks from the odd ranks
        {
            // determine target-rank to recv from
            int recv_from_rank = my_rank + 1;
            int mpi_tag = recv_from_rank;

            // recieving tasks
            void *buffer = malloc(recv_buff_size);
            std::vector<mig_task_t *> recv_task_list;
            recv_tasks_from_rank(buffer, mpi_tag, recv_from_rank, recv_buff_size, recv_task_list);

        } else {

            // determine target-rank to send
            int send_to_rank = my_rank - 1;
            int num_tasks = 1;

            // create tasks
            mig_task_t **task_ptr = (mig_task_t**) malloc(sizeof(mig_task_t *) * num_tasks);
            for (int i = 0; i < num_tasks; i++){
                mig_task_t task = mig_task_t(my_rank, i, 5);
                mig_task_t *ptr = &task;
                task_ptr[i] = ptr;
            }

            // send tasks
            offload_tasks_to_rank(my_rank, task_ptr, num_tasks, send_to_rank);
        }
        timer_2side_comm.pauseTime();
    }
    double throughput_2side_comm = (num_request*recv_buff_size*1000) / (timer_2side_comm.getElapsedTime()*1024*1024);

    // printf("CHECK_TIMER: R%d elapsed time = %f\n", my_rank, timer_2side_comm.getElapsedTime()/1000);
    printf("Total throughput_2side_comm: %f (MB/s)\n", throughput_2side_comm);

    MPI_Finalize();
    
    exit(EXIT_SUCCESS);
}
