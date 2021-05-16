#include <algorithm>

#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>
#include <bcl/containers/CircularQueue.hpp>

#define NUM_TASKS 100

/**
 * A simple task struct for testing
 */
typedef struct task_t
{
    int tid;
    double  A[100];
    double  B[100];
    double  C[100];
} task_t;

int main(int argc, char** argv) {

    /* init BCL env */
    BCL::init();

    /* declare queue size (num of elements in queue) */
    size_t queue_size = NUM_TASKS * 2;
    std::vector<BCL::CircularQueue<task_t>> bcl_c_queue;
    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
        bcl_c_queue.push_back(BCL::CircularQueue<task_t>(rank, queue_size));
    }

    /* Init tasks and push to the queue */
    srand48(BCL::rank());
    for (size_t i = 0; i < NUM_TASKS; i++) {
        
        size_t dst_rank = lrand48() % BCL::nprocs();

        // init info for each task
        task_t tmp;
        tmp.tid = dst_rank*NUM_TASKS + i;    // just for make different tids per rank
        for (int j = 0;  j < 100; j++){
            tmp.A[j] = 1;
            tmp.B[j] = 2;
            tmp.C[j] = 3;
        }

        BCL::print("[PUSH] R%d pushes Task-%d to the queue...\n", dst_rank, tmp.tid);
        bcl_c_queue[dst_rank].push(tmp);
    }

    BCL::barrier();

    // Sort local queue in place
    // std::sort(bcl_c_queue[BCL::rank()].begin().local(), bcl_f_queue[BCL::rank()].end().local());
    BCL::print("-----------------------------------------\n");

    /* Pop tasks out of the queue */
    size_t count = 0;
    while (!bcl_c_queue[BCL::rank()].empty()) {
        task_t t_value;

        bool success = bcl_c_queue[BCL::rank()].pop(t_value);

        BCL::print("[POP] Task %d: from Rank %d...\n", t_value.tid, BCL::rank());

        if (success) {
            count++;
        }
    }

    size_t total_count = BCL::allreduce<size_t>(count, std::plus<size_t>{});
    BCL::print("Done! Popped %lu values total.\n", total_count);

    BCL::finalize();
    return 0;
}
