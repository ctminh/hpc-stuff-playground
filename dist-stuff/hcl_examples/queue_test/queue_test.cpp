#include <sys/types.h>
#include <unistd.h>

#include <functional>
#include <utility>
#include <mpi.h>
#include <iostream>
#include <signal.h>
#include <execinfo.h>
#include <chrono>
#include <queue>
#include <fstream>
#include <hcl/common/data_structures.h>
#include <hcl/queue/queue.h>

const int MAT_SIZE = 128;

/* Try a complex type as the elements in hcl-queue */
struct DoubleType {

    std::array<double, MAT_SIZE*MAT_SIZE> A;
    // std::array<double, MAT_SIZE*MAT_SIZE> B;
    // std::array<double, MAT_SIZE*MAT_SIZE> C;

    // constructor 1
    DoubleType() : A() {}
    // DoubleType() : A(), B(), C() {}
    // DoubleType() { A.fill(1.0); B.fill(2.0); C.fill(0.0); }

    // constructor 2
    // DoubleType(int val) : A(MAT_SIZE * MAT_SIZE, val), B(MAT_SIZE * MAT_SIZE, val), C(MAT_SIZE * MAT_SIZE, val) {}
    DoubleType(std::array<double, MAT_SIZE*MAT_SIZE> a_) : A(a_) {}
    // DoubleType(std::array<double, MAT_SIZE*MAT_SIZE> a_, std::array<double, MAT_SIZE*MAT_SIZE> b_, std::array<double, MAT_SIZE*MAT_SIZE> c_) :
    //     A(a_), B(b_), C(c_) {}

    // constructor 3
    DoubleType(int val) {
        for (int i = 0; i < A.size(); i++){
            A[i] = double(val);
            // B[i] = double(val);
            // C[i] = 0.0;
        }
    }

    /* Equal operator for comparing two Matrix */
    bool operator==(const DoubleType &o) const {
        if (o.A.size() != A.size()) return false;
        // if (o.B.size() != B.size()) return false;
        // if (o.C.size() != C.size()) return false;
        for (int i = 0; i < A.size(); ++i) {
            if (o.A[i] != A[i]) return false;
            // if (o.B[i] != B[i]) return false;
            // if (o.C[i] != C[i]) return false;
        }
        return true;
    }

    DoubleType &operator=(const DoubleType &other) {
        A = other.A;
        // B = other.B;
        // C = other.C;
        return *this;
    }

    bool operator<(const DoubleType &o) const {
        if (o.A.size() < A.size()) return false;
        if (o.A.size() > A.size()) return true;
        // if (o.B.size() < B.size()) return false;
        // if (o.B.size() > B.size()) return true;
        // if (o.C.size() < C.size()) return false;
        // if (o.C.size() > C.size()) return true;
        for (int i = 0; i < A.size(); ++i) {
            if (o.A[i] < A[i]) return false;
            if (o.A[i] > A[i]) return true;
            // if (o.B[i] < B[i]) return false;
            // if (o.B[i] > B[i]) return true;
            // if (o.C[i] < C[i]) return false;
            // if (o.C[i] > C[i]) return true;
        }
        return false;
    }

    bool operator>(const DoubleType &o) const {
        // return !(*this < o);
        return !(A < o.A);
    }

    bool Contains(const DoubleType &o) const {
        // return *this == o;
        return A == o.A;
    }

};

// serialization
#if defined(HCL_ENABLE_THALLIUM_TCP) || defined(HCL_ENABLE_THALLIUM_ROCE)
    template<typename A>
    void serialize(A &ar, DoubleType &a) {
        ar & a.A;
        // ar & a.B;
        // ar & a.C;
    }
#endif

namespace std {
    template<>
    struct hash<DoubleType> {
        size_t operator()(const DoubleType &k) const {
            size_t hash_val = hash<int>()(k.A[0]);
            for (int i = 1; i < k.A.size(); ++i) {
                hash_val ^= hash<int>()(k.A[0]);
            }
            return hash_val;
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
    int comm_size,my_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    int ranks_per_server=comm_size,num_request=100;
    long size_of_request=1000;
    bool debug=false;
    bool server_on_node=false;
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
    
    if(debug && my_rank==0){
        printf("%d ready for attach\n", comm_size);
        fflush(stdout);
        getchar();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    bool is_server=(my_rank+1) % ranks_per_server == 0;
    int my_server=my_rank / ranks_per_server;
    int num_servers=comm_size/ranks_per_server;

    // The following is used to switch to 40g network on Ares.
    // This is necessary when we use RoCE on Ares.
    std::string proc_name = std::string(processor_name);
    /* int split_loc = proc_name.find('.');
    std::string node_name = proc_name.substr(0, split_loc);
    std::string extra_info = proc_name.substr(split_loc+1, string::npos);
    proc_name = node_name + "-40g." + extra_info; */

    size_t size_of_elem = sizeof(double);
    const int array_size = 3*MAT_SIZE*MAT_SIZE;
    // const int array_size = TEST_REQUEST_SIZE;
    // if (size_of_request != array_size) {
    //     printf("Please set TEST_REQUEST_SIZE in include/hcl/common/constants.h instead. Testing with %d\n", array_size);
    // }
    std::array<int,array_size> my_vals=std::array<int,array_size>();

    
    HCL_CONF->IS_SERVER = is_server;
    HCL_CONF->MY_SERVER = my_server;
    HCL_CONF->NUM_SERVERS = num_servers;
    HCL_CONF->SERVER_ON_NODE = server_on_node || is_server;
    HCL_CONF->SERVER_LIST_PATH = "./server_list";

    auto mem_size = MAT_SIZE*MAT_SIZE * (comm_size + 1)*num_request;
    HCL_CONF->MEMORY_ALLOCATED = mem_size;
    printf("Rank Config %d %d %d %d %d %lu, size_elem=%ld, my_vals=%ld\n", my_rank, HCL_CONF->IS_SERVER, HCL_CONF->MY_SERVER, HCL_CONF->NUM_SERVERS,
                HCL_CONF->SERVER_ON_NODE, HCL_CONF->MEMORY_ALLOCATED, size_of_elem, my_vals.size());

    hcl::queue<DoubleType> *queue;
    if (is_server) {
        queue = new hcl::queue<DoubleType>();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!is_server) {
        queue = new hcl::queue<DoubleType>();
    }

    std::queue<DoubleType> lqueue=std::queue<DoubleType>();

    MPI_Comm client_comm;
    MPI_Comm_split(MPI_COMM_WORLD, !is_server, my_rank, &client_comm);
    int client_comm_size;
    MPI_Comm_size(client_comm, &client_comm_size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (!is_server) {
        Timer llocal_queue_timer=Timer();
        std::hash<DoubleType> keyHash;

        /*Local std::queue test*/
        for(int i=0;i<num_request;i++){
            double val = my_server;
            llocal_queue_timer.resumeTime();
            lqueue.push(DoubleType(val));
            llocal_queue_timer.pauseTime();
        }

        double llocal_queue_throughput=num_request/llocal_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        Timer llocal_get_queue_timer=Timer();
        for(int i=0;i<num_request;i++){
            double val = my_server;
            llocal_get_queue_timer.resumeTime();
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
        printf("[DBG-REMOTEQUEUE-LOCAL-OPS] R%d: my_server_key = %d\n", my_rank, my_server_key);

        /* Local queue test */
        for(int i=0;i<num_request;i++){
            double val = my_server;
            auto key=DoubleType(val);
            local_queue_timer.resumeTime();
            queue->Push(key, my_server_key);
            local_queue_timer.pauseTime();
        }
        double local_queue_throughput=num_request/local_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        Timer local_get_queue_timer=Timer();
        for(int i=0;i<num_request;i++){
            double val = my_server;
            auto key=DoubleType(val);
            local_get_queue_timer.resumeTime();
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

        /* Remote queue test */
        Timer remote_queue_timer=Timer();
        uint16_t my_server_remote_key = (my_server + 1) % num_servers;
        uint16_t my_server_local_key = my_server % num_servers;;
        for(int i=0;i<num_request;i++){
            double val = my_server+1;
            auto key=DoubleType(val);
            remote_queue_timer.resumeTime();
            queue->Push(key, my_server_local_key);
            remote_queue_timer.pauseTime();
        }
        double remote_queue_throughput=num_request/remote_queue_timer.getElapsedTime()*1000*size_of_elem*my_vals.size()/1024/1024;

        MPI_Barrier(client_comm);

        Timer remote_get_queue_timer=Timer();
        for(int i=0;i<num_request;i++){
            double val = my_server+1;
            auto key=DoubleType(val);
            remote_get_queue_timer.resumeTime();
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