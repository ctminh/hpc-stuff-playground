#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <queue>
#include <mpi.h>
#include <hcl/common/data_structures.h>
#include <hcl/queue/queue.h>

struct KeyType{
    size_t a;

    // constructor 1
    KeyType():a(0){ }

    // constructor 2
    KeyType(size_t a_):a(a_){ }

    #ifdef HCL_ENABLE_RPCLIB
        MSGPACK_DEFINE(a);
    #endif

    /* equal operator for comparing two Matrix */
    bool operator == (const KeyType &o) const {
        return a == o.a;
    }

    KeyType& operator = (const KeyType& other) {
        a = other.a;
        return *this;
    }

    bool operator < (const KeyType &o) const {
        return a < o.a;
    }

    bool operator > (const KeyType &o) const {
        return a > o.a;
    }

    bool Contains(const KeyType &o) const {
        return a == o.a;
    }

    #if defined(HCL_ENABLE_THALLIUM_TCP) || defined(HCL_ENABLE_THALLIUM_ROCE)
        template<typename A>
        void serialize(A& ar) const {
            ar & a;
        }
    #endif
};

namespace std {
    template <>
    struct hash<KeyType> {
        size_t operator () (const KeyType &k) const {
            return k.a;
        }
    };
}


/* main function */
int main(int argc, char * argv[])
{
    // init MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    printf("[DEBUG] provided = %d; MPI_THREAD_MULTIPLE = %d\n", provided, MPI_THREAD_MULTIPLE);
    if (provided < MPI_THREAD_MULTIPLE) {
        printf("Didn't receive appropriate MPI threading specification\n");
        exit(EXIT_FAILURE);
    }

    // some MPI declaration
    int comm_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int ranks_per_server = comm_size;
    int num_request = 10000;
    long size_of_request = 1000;
    bool debug = false;
    bool server_on_node = false;
    if(argc > 1)    ranks_per_server    = atoi(argv[1]);
    if(argc > 2)    num_request         = atoi(argv[2]);
    if(argc > 3)    size_of_request     = (long)atol(argv[3]);
    if(argc > 4)    server_on_node      = (bool)atoi(argv[4]);
    if(argc > 5)    debug               = (bool)atoi(argv[5]);

    // get compute_node names
    int len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &len);
    if (debug) {
        printf("[DEBUG] %s/%d: %d\n", processor_name, rank, getpid());
    }
    if (debug && rank == 0){
        printf("[DEBUG] %d ready for attaching\n", comm_size);
        fflush(stdout);
        getchar();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // who is the server?
    bool is_server = (rank == 0); // (rank + 1) % ranks_per_server == 0;
    size_t my_server = 0; // rank / ranks_per_server;
    int num_servers = comm_size / ranks_per_server;

    // for debugging
    std::string proc_name = std::string(processor_name);
    size_t size_of_elem = sizeof(int);
    printf("[DEBUG] Rank %d, (is_server? %d), my_server = %zu, num_servers = %d\n", rank, is_server, my_server, num_servers);

    const int array_size = TEST_REQUEST_SIZE;
    if (size_of_request != array_size) {
        printf("Please set TEST_REQUEST_SIZE in include/hcl/common/constants.h instead. Testing with %d\n", array_size);
    }

    // create an array for testing
    std::array<int, array_size> my_vals = std::array<int, array_size>();

    // config HCL containers
    HCL_CONF->IS_SERVER = is_server;
    HCL_CONF->MY_SERVER = my_server;
    HCL_CONF->NUM_SERVERS = num_servers;
    HCL_CONF->SERVER_ON_NODE = server_on_node || is_server;
    HCL_CONF->SERVER_LIST_PATH = "./server_list";

    // create HCL queue
    hcl::queue<KeyType> *queue;
    if (is_server){
        queue = new hcl::queue<KeyType>();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!is_server){
        queue = new hcl::queue<KeyType>();
    }

    // create a local queue
    std::queue<KeyType> lqueue = std::queue<KeyType>(); // a queue with the type is KeyType
    MPI_Comm client_comm;   // to determine which processes are involed
    // create a new communicator based on colors and keys
    // color is !is_server, key is rank
    MPI_Comm_split(MPI_COMM_WORLD, !is_server, rank, &client_comm);
    int client_comm_size;
    MPI_Comm_size(client_comm, &client_comm_size);
    printf("[DEBUG] Difference between client_comm_size vs comm_size: %d vs %d\n", client_comm_size, comm_size);
    MPI_Barrier(MPI_COMM_WORLD);

    // if not the server
    if (!is_server) {
        Timer llocal_queue_timer = Timer();
        std::hash<KeyType> keyHash;

        ///////////////////////////////////////////////////////
        ///////////// llocal queue test - push ////////////////
        for (int i = 0; i < num_request; i++) {
            size_t val = my_server;
            llocal_queue_timer.resumeTime();
            size_t key_hash = keyHash(KeyType(val)) % num_servers;
            if (key_hash == my_server && is_server){
                // do nothing
            }
            lqueue.push(KeyType(val));
            llocal_queue_timer.pauseTime();
        }

        double llocal_queue_throughput = num_request / llocal_queue_timer.getElapsedTime() * 1000 * size_of_elem * my_vals.size() / 1024 / 1024;

        // llocal queue test: pop
        Timer llocal_get_queue_timer = Timer();
        for (int i = 0; i < num_request; i++) {
            size_t val = my_server;
            llocal_queue_timer.resumeTime();
            size_t key_hash = keyHash(KeyType(val)) % num_servers;
            if (key_hash == my_server && is_server){
                // do nothing
            }
            auto result = lqueue.front();
            lqueue.pop();
            llocal_get_queue_timer.pauseTime();
        }

        double llocal_get_queue_throughput = num_request / llocal_get_queue_timer.getElapsedTime() * 1000 * size_of_elem * my_vals.size() / 1024 / 1024;

        if (rank == 0) {
            printf("llocal_queue_throughput put: %f\n",llocal_queue_throughput);
            printf("llocal_queue_throughput get: %f\n",llocal_get_queue_throughput);
        }
        MPI_Barrier(client_comm);


        ///////////////////////////////////////////////////////
        ///////////// local queue test - push /////////////////
        Timer local_queue_timer = Timer();
        uint16_t my_server_key = my_server % num_servers;
        for(int i = 0; i < num_request; i++){
            size_t val = my_server;
            auto key = KeyType(val);
            local_queue_timer.resumeTime();
            queue->Push(key, my_server_key);
            local_queue_timer.pauseTime();
        }
        double local_queue_throughput = num_request / local_queue_timer.getElapsedTime() * 1000 * size_of_elem * my_vals.size() / 1024 / 1024;

        // local queue test - pop
        Timer local_get_queue_timer = Timer();
        for(int i = 0; i < num_request; i++){
            size_t val = my_server;
            auto key = KeyType(val);
            local_get_queue_timer.resumeTime();
            size_t key_hash = keyHash(key) % num_servers;
            if (key_hash == my_server && is_server){
                // do nothing
            }
            auto result = queue->Pop(my_server_key);
            local_get_queue_timer.pauseTime();
        }
        double local_get_queue_throughput = num_request / local_get_queue_timer.getElapsedTime() * 1000 * size_of_elem * my_vals.size() / 1024 / 1024;

        double local_put_tp_result, local_get_tp_result;
        if (client_comm_size > 1) {
            MPI_Reduce(&local_queue_throughput, &local_put_tp_result, 1, MPI_DOUBLE, MPI_SUM, 0, client_comm);
            MPI_Reduce(&local_get_queue_throughput, &local_get_tp_result, 1, MPI_DOUBLE, MPI_SUM, 0, client_comm);
            local_put_tp_result /= client_comm_size;
            local_get_tp_result /= client_comm_size;
        }
        else {
            local_put_tp_result = local_queue_throughput;
            local_get_tp_result = local_get_queue_throughput;
        }

        if (rank == 0) {
            printf("local_queue_throughput put: %f\n", local_put_tp_result);
            printf("local_queue_throughput get: %f\n", local_get_tp_result);
        }

        MPI_Barrier(client_comm);


        ///////////////////////////////////////////////////////
        ///////////// remote queue test - push ////////////////
        Timer remote_queue_timer = Timer();
        uint16_t my_server_remote_key = (my_server + 1) % num_servers;
        for (int i = 0; i < num_request; i++){
            size_t val = my_server + 1;
            auto key = KeyType(val);
            remote_queue_timer.resumeTime();
            queue->Push(key, my_server_remote_key);
            remote_queue_timer.pauseTime();
        }
        // remote-queue push-throughput
        double remote_queue_throughput = num_request / remote_queue_timer.getElapsedTime() * 1000 * size_of_elem * my_vals.size() / 1024 / 1024;

        MPI_Barrier(client_comm);

        // remote queue test - pop
        Timer remote_get_queue_timer = Timer();
        for(int i = 0 ; i < num_request; i++){
            size_t val = my_server + 1;
            auto key = KeyType(val);
            remote_get_queue_timer.resumeTime();
            size_t key_hash = keyHash(key) % num_servers;
            if (key_hash == my_server && is_server){
                // do nothing
            }
            queue->Pop(my_server_remote_key);
            remote_get_queue_timer.pauseTime();
        }
        // remote-queue pop-throughput
        double remote_get_queue_throughput = num_request / remote_get_queue_timer.getElapsedTime() * 1000 * size_of_elem * my_vals.size() / 1024 / 1024;

        // get some benchmark results
        double remote_put_tp_result, remote_get_tp_result;
        if (client_comm_size > 1){
            MPI_Reduce(&remote_queue_throughput, &remote_put_tp_result, 1, MPI_DOUBLE, MPI_SUM, 0, client_comm);
            remote_put_tp_result /= client_comm_size;
            MPI_Reduce(&remote_get_queue_throughput, &remote_get_tp_result, 1, MPI_DOUBLE, MPI_SUM, 0, client_comm);
            remote_get_tp_result /= client_comm_size;
        }
        else
        {
            remote_put_tp_result = remote_queue_throughput;
            remote_get_tp_result = remote_get_queue_throughput;
        }

        if(rank == 0) {
            printf("remote queue throughput (put): %f\n", remote_put_tp_result);
            printf("remote queue throughput (get): %f\n", remote_get_tp_result);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    delete(queue);

    MPI_Finalize();

    exit(EXIT_SUCCESS);

    return 0;
}