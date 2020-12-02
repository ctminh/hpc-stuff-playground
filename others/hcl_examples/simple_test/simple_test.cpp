#include <hcl.h>
#include <iostream>
#include <mpi.h>

int main(int argc, char **argv) {
    // init MPI
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // rank 0
    int my_server = 0;
    hcl::queue<int> int_queue("QUEUE", rank == my_server, my_server, 1, true);
    if (rank == my_server) {
        int_queue.WaitForElement(my_server);
        auto result = int_queue.Pop(my_server);
        if (result.first) {
            std::cout << result.second << std::endl;
        }
    } else {    // other ranks
        int_queue.Push(42, my_server);
    }

    MPI_Finalize();

    return 0;
}