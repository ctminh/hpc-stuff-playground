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
    mig_task_t()
    {
        id = 1;
        idx_image = 1;
        arg_num = 5;
        void *ptr = nullptr;
        arg_hst_pointers.push_back(ptr);
        arg_sizes.push_back(10240);
        arg_sizes.push_back(20480);
        arg_sizes.push_back(40960);
        arg_sizes.push_back(51200);
        arg_sizes.push_back(102400);
        arg_sizes.push_back(204800);
    }

    // constructor 2
    mig_task_t(int32_t img, int task_id, int32_t num_args)
    {
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

void offload_action(mig_task_t *task, int target_rank, bool use_synchronous_mode) {
    // encode buffer before sending tasks
    int32_t buffer_size = 0;
    void *buffer = NULL;
    int num_bytes_sent = 0;

    buffer = encode_send_buffer(tasks, &buffer_size);

    MPI_Request *requests = new MPI_Request[n_requests];
    #if MPI_BLOCKING
        MPI_Send(buffer, buffer_size, MPI_BYTE, target_rank, tmp_tag, chameleon_comm);
    #else
    if(use_synchronous_mode)
        MPI_Issend(buffer, buffer_size, MPI_BYTE, target_rank, tmp_tag, chameleon_comm, &requests[0]);
    else
        MPI_Isend(buffer, buffer_size, MPI_BYTE, target_rank, tmp_tag, chameleon_comm, &requests[0]);
    #endif

#if OFFLOAD_DATA_PACKING_TYPE > 0 && CHAM_STATS_RECORD
    cur_time = omp_get_wtime();
#endif

#if OFFLOAD_DATA_PACKING_TYPE == 1
    int cur_req_index = 1;
#if CHAM_STATS_RECORD
    int tmp_bytes_send = 0;
#endif
    for(int i_task = 0; i_task < num_tasks; i_task++) {
        cham_migratable_task_t *task = tasks[i_task];
        for(int i=0; i<task->arg_num; i++) {
            int is_lit      = task->arg_types[i] & CHAM_OMP_TGT_MAPTYPE_LITERAL;
            int is_to       = task->arg_types[i] & CHAM_OMP_TGT_MAPTYPE_TO;
            if(is_to) {
                if(is_lit) {
                    #if MPI_BLOCKING
                    MPI_Send(&task->arg_hst_pointers[i], task->arg_sizes[i], MPI_BYTE, target_rank, tmp_tag, chameleon_comm);
                    #else
                    if(use_synchronous_mode)
                        MPI_Issend(&task->arg_hst_pointers[i], task->arg_sizes[i], MPI_BYTE, target_rank, tmp_tag, chameleon_comm, &requests[cur_req_index]);
                    else
                        MPI_Isend(&task->arg_hst_pointers[i], task->arg_sizes[i], MPI_BYTE, target_rank, tmp_tag, chameleon_comm, &requests[cur_req_index]);
                    #endif                
                }
                else {
                    #if MPI_BLOCKING
                    MPI_Send(task->arg_hst_pointers[i], task->arg_sizes[i], MPI_BYTE, target_rank, tmp_tag, chameleon_comm);
                    #else
                    if(use_synchronous_mode)
                        MPI_Issend(task->arg_hst_pointers[i], task->arg_sizes[i], MPI_BYTE, target_rank, tmp_tag, chameleon_comm, &requests[cur_req_index]);
                    else
                        MPI_Isend(task->arg_hst_pointers[i], task->arg_sizes[i], MPI_BYTE, target_rank, tmp_tag, chameleon_comm, &requests[cur_req_index]);
                    #endif
                }
#if CHAM_STATS_RECORD
                tmp_bytes_send += task->arg_sizes[i];
                _stats_bytes_send_per_message.add_stat_value((double)task->arg_sizes[i]);
#endif
                cur_req_index++;
                print_arg_info("offload_action - sending argument", task, i);
            }
        }
    }
#if CHAM_STATS_RECORD
    num_bytes_sent += tmp_bytes_send;
    cur_time = omp_get_wtime()-cur_time;
    #if MPI_BLOCKING
    add_throughput_send(cur_time, tmp_bytes_send);
    #endif
#endif

#elif OFFLOAD_DATA_PACKING_TYPE == 2
    int tmp_overall_arg_nums = 0;
    for(int i_task = 0; i_task < num_tasks; i_task++) {
        for(int tmp_i_arg = 0; tmp_i_arg < tasks[i_task]->arg_num; tmp_i_arg++) {
            int is_to = tasks[i_task]->arg_types[tmp_i_arg] & CHAM_OMP_TGT_MAPTYPE_TO;
            if(is_to)
                tmp_overall_arg_nums++;
        }
    }

    MPI_Datatype type_mapped_vars;
    MPI_Datatype separate_types[tmp_overall_arg_nums];
    int blocklen[tmp_overall_arg_nums];
    MPI_Aint disp[tmp_overall_arg_nums];
    int ierr = 0;
    int tmp_count = 0;

    for(int i_task = 0; i_task < num_tasks; i_task++) {
        cham_migratable_task_t *task = tasks[i_task];

        for(int i=0; i<task->arg_num; i++) {
            int is_to                       = task->arg_types[i] & CHAM_OMP_TGT_MAPTYPE_TO;
            if(is_to) {
                separate_types[tmp_count]   = MPI_BYTE;
                blocklen[tmp_count]         = task->arg_sizes[i];
                int is_lit                  = task->arg_types[i] & CHAM_OMP_TGT_MAPTYPE_LITERAL;
                
                if(is_lit) {
                    ierr = MPI_Get_address(&task->arg_hst_pointers[i], &(disp[tmp_count]));
                }
                else {
                    ierr = MPI_Get_address(task->arg_hst_pointers[i], &(disp[tmp_count]));
                }
                tmp_count++;
            }
        }
    }

    ierr = MPI_Type_create_struct(tmp_overall_arg_nums, blocklen, disp, separate_types, &type_mapped_vars);
    assert(ierr==MPI_SUCCESS);
    ierr = MPI_Type_commit(&type_mapped_vars);
    assert(ierr==MPI_SUCCESS);

#if CHAM_STATS_RECORD
    int size = 0;
    MPI_Type_size(type_mapped_vars, &size);
    num_bytes_sent += size;
    _stats_bytes_send_per_message.add_stat_value((double)size);
#endif
    #if MPI_BLOCKING
    ierr = MPI_Send(MPI_BOTTOM, 1, type_mapped_vars, target_rank, tmp_tag, chameleon_comm);
    #else
    if( use_synchronous_mode )
        ierr = MPI_Issend(MPI_BOTTOM, 1, type_mapped_vars, target_rank, tmp_tag, chameleon_comm, &requests[1]);
    else
        ierr = MPI_Isend(MPI_BOTTOM, 1, type_mapped_vars, target_rank, tmp_tag, chameleon_comm, &requests[1]);
    #endif

    assert(ierr==MPI_SUCCESS);

#if CHAM_STATS_RECORD
    cur_time = omp_get_wtime()-cur_time;
    #if MPI_BLOCKING
    add_throughput_send(cur_time, size);
    #endif
#endif

    ierr = MPI_Type_free(&type_mapped_vars);
    assert(ierr==MPI_SUCCESS);
#endif /* OFFLOAD_DATA_PACKING_TYPE */

    for(int i_task = 0; i_task < num_tasks; i_task++) {
        cham_migratable_task_t *task = tasks[i_task];
        if(task->HasAtLeastOneOutput()) {
            _tasks_to_deallocate.push_back(task);
            _map_offloaded_tasks_with_outputs.insert(task->task_id, task);
            DBP("offload_action - inserted task with id %ld and pointer %p into offloaded map with outputs\n", task->task_id, task);
            assert(task->num_outstanding_recvbacks>=0);
            task->num_outstanding_recvbacks++;
            DBP("offload_action - increment outstanding recvbacks for task with id %ld new count: %d\n", task->task_id, task->num_outstanding_recvbacks);
            assert(task->num_outstanding_recvbacks>0);

            // early irecv here
            #if CHAM_REPLICATION_MODE==0 && ENABLE_EARLY_IRECVS==1
            if(!task->is_replicated_task) {
                action_post_recvback_requests(task, target_rank, task->task_id, &request_manager_receive);
            }
            #endif
        }
    }
    _active_migrations_per_target_rank[target_rank]++;

    #if MPI_BLOCKING
    send_handler(buffer, tmp_tag, target_rank, nullptr, 0);
    #else
    request_manager_send.submitRequests(start_time_requests, tmp_tag, target_rank, n_requests, 
                                requests,
                                num_bytes_sent,
                                0,
                                send_handler,
                                send,
                                buffer,
                                tasks,
                                num_tasks);
    #endif
    delete[] requests;
    DBP("offload_action (exit)\n");
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

    printf("rank %d, is_server %d, my_server %zu, num_servers %d\n", my_rank, is_server, my_server, num_servers);

    HCL_CONF->IS_SERVER = is_server;
    HCL_CONF->MY_SERVER = my_server;
    HCL_CONF->NUM_SERVERS = num_servers;
    HCL_CONF->SERVER_ON_NODE = server_on_node || is_server;
    HCL_CONF->SERVER_LIST_PATH = "./server_list";

    // create hcl_queue with type is ChamTaskType
    hcl::queue<mig_task_t> *glob_queue;
    if (is_server) {
        glob_queue = new hcl::queue<mig_task_t>();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!is_server) {
        glob_queue = new hcl::queue<mig_task_t>();
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
            mig_task_t task = mig_task_t(1, i, 5);
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
            mig_task_t task = mig_task_t(1, i, 5);
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


    /* ///////////////// Test 2-Sided Communication  ////////////////////////////// */
    /* ///////////////// Paired_Process for Send/Recv Tasks       ///////////////// */



    MPI_Finalize();
    
    exit(EXIT_SUCCESS);
}
