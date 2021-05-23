/*
 * This example is refered from the repo of Nikola, TUM. He developed and
 * modified something in BCL-src. The original source code is here:
 * https://github.com/ge49nuk/BCLmatrix/tree/master/examples/simple
 * 
 */

#include <bcl/bcl.hpp>
#include <bcl/containers/CircularQueue.hpp>
#include <bcl/containers/FastQueue.hpp>
#include <string>
#include <cstdlib>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <bitset>
#include <random>

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

// ================================================================================
// Global Variables
// ================================================================================
using namespace std;

bool verifyResults = false;
bool ShowRuntimeDist = true;
double stealPerc, stealTime = 0, multiplyTime = 0, responseTime = 0;

// for tracking num tasks that are stolen
std::vector<int> num_stolen_tasks;

/* For matrix multiplication and initialization */
void multiply(double *matrix, double *matrix2, double *result);
void initialize_matrix_rnd(double *mat);
bool steal(std::vector<BCL::CircularQueue<int>> *queues);

// ================================================================================
// Util Functions
// ================================================================================

/* For measuring time */
static inline double curtime(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

double execute(int tasks, int taskStealing){

    double fTimeStart, fTimeEnd, elapsed = 0;

    if (BCL::rank() == 0){
        printf("Executing code with: RANKS:%ld  THREADS:%d  MSIZE:%d  STEALP:%lf  STEALING:%d\n",
                BCL::nprocs(), omp_get_max_threads(), MSIZE, stealPerc, taskStealing);
    }

    // the queues, which is created by CircularQueue<int>, just holds the task id
    std::vector<BCL::CircularQueue<int>> queues;

    // taskPointerArray is an array that holds the global pointer
    std::vector<BCL::Array<BCL::GlobalPtr<task>>> taskPointerArray;

    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
        queues.push_back(BCL::CircularQueue<int>(rank, 30000));
        taskPointerArray.push_back(BCL::Array<BCL::GlobalPtr<task>>(rank, 1));
    }

    taskPointerArray[BCL::rank()][0].put(BCL::alloc<task>(tasks));

    //create and save tasks
    int nextId = 0;
    for (int i = 0; i < tasks; i++) {
        task t;
        t.matrixSize = MSIZE;
        t.taskId = (int)(BCL::rank());
        t.taskId <<= 24;
        t.taskId += nextId;

        initialize_matrix_rnd(t.matrix);
        initialize_matrix_rnd(t.matrix2);

        BCL::memcpy(taskPointerArray[BCL::rank()][0].get() + nextId, &t, sizeof(task));

        queues[BCL::rank()].push(t.taskId, BCL::CircularQueueAL::push);

        nextId++;
    }
    BCL::barrier();

    // create globel task pointers
    vector<BCL::GlobalPtr<task>> taskPtrs;
    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
        taskPtrs.push_back(taskPointerArray[rank][0].get());
    }

    // measure time for solving tasks and save the result at the origin rank
    fTimeStart = curtime();

    // the main loop for executing tasks
    while (true) {

        // measure just the time for computing mxm-multiplication
        double startMult = curtime();

        #pragma omp parallel for
        for (int i = 0; i < queues[BCL::rank()].size(); i++) {

            // pop tasks with the task id, i.e., tId
            int tId;
            bool success = queues[BCL::rank()].pop(tId);

            if (success) {
                
                // determine the position of the task from tId
                int ownerRank = tId >> 24;
                int taskIndex = (tId << 8) >> 8;

                // get the task from the remote rank
                task t;
                BCL::memcpy(&t, taskPtrs[ownerRank] + taskIndex, sizeof(task));

                // cal mxm-multiplication kernel
                multiply(t.matrix, t.matrix2, t.result);

                // calculate the offset of the task at the remote side
                int offSet = 2 * MSIZE * MSIZE * sizeof(double) + sizeof(int);
                BCL::memcpy(taskPtrs[ownerRank] + taskIndex, (&t), sizeof(task));
            }
        }
        multiplyTime += curtime() - startMult;

        // check the queues
        if (queues[BCL::rank()].size() == 0 && (!taskStealing || !steal(&queues))) {
            break;
        }
    }

    // calculate the elapsed time and show it
    elapsed = curtime() - fTimeStart;
    if (ShowRuntimeDist){
        printf("[R%ld] Steal Time: %.3f %%, Multiply time: %.3f %%, Response time: %.3f %% | Num Stolen-Tasks: %d\n",
                BCL::rank(), stealTime / elapsed * 100, multiplyTime / elapsed * 100, responseTime / elapsed * 100, num_stolen_tasks[BCL::rank()]);
    }
    BCL::barrier();

    if (BCL::rank() == 0) {
        fTimeEnd = curtime();
        elapsed = fTimeEnd - fTimeStart;
        printf("%lf\n", elapsed);
    }

    // verify the mxm-multiplication results
    task t;
    if (verifyResults){
        for (int i = 0; i < tasks; i++) {
            t = taskPtrs[BCL::rank()][i];
            double *result = (double *)malloc(MSIZE * MSIZE * sizeof(double));
            multiply(t.matrix, t.matrix2, result);

            for (int i = 0; i < MSIZE * MSIZE; i++) {
                if (result[i] != t.result[i]) {
                    printf("[R%ld] The multiplication of task with TID: %d was computed incorrectly!\n", BCL::rank(), t.taskId);
                    break;
                }
            }
            free(result);
        }
    }

    BCL::barrier();

    return elapsed;
}

void initialize_matrix_rnd(double *mat) {
    double lower_bound = 0;
    double upper_bound = 10;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    for (int i = 0; i < MSIZE * MSIZE; i++) {
        mat[i] = unif(re);
    }
}

void multiply(double *matrix, double *matrix2, double *result) {

    // put itac trace here
#if TRACE==1
    static int event_multiply = -1;
    std::string event_multiply_name = "multiply";
    if(event_multiply == -1) 
        int ierr = VT_funcdef(event_multiply_name.c_str(), VT_NOCLASS, &event_multiply);
    VT_BEGIN_CONSTRAINED(event_multiply);
#endif

    for (int i = 0; i < MSIZE * MSIZE; i += 1) {
        double value = 0;
        int k = i % MSIZE;
        for (int j = (i / MSIZE) * MSIZE; j < (i / MSIZE) * MSIZE + MSIZE; j++) {
            value = value + matrix[j] * matrix2[k];
            k += MSIZE;
        }
        result[i] = value;
    }

#if TRACE==1
    VT_END_W_CONSTRAINED(event_multiply);
#endif

}

bool steal(std::vector<BCL::CircularQueue<int>> *queues) {

    std::srand(unsigned(std::time(0)));
    std::vector<int> ranks;

    for (int i = 0; i < BCL::nprocs(); i++) {
        if (i != BCL::rank())
            ranks.push_back(i);
    }

    std::random_shuffle(ranks.begin(), ranks.end());

    // iterates through ranks and tests if they have tasks left
    for (std::vector<int>::iterator it = ranks.begin(); it != ranks.end(); ++it) {

        //measuring the time until the first response from foreign rank
        double start = curtime();
        long size = (*queues)[*it].size();
        responseTime += curtime() - start;

        int minAmount = omp_get_max_threads() - 1;
        if (size > 0) { // steals half the tasks
            int j = 0;
            while (j < (*queues)[*it].size() * stealPerc) {
                int tId;
                start = curtime();
                if ((*queues)[*it].pop(tId)) {

                    // try to put a trace-event here
                    #if TRACE==1
                    static int event_steal = -1;
                    std::string event_steal_name = "steal";
                    if(event_steal == -1) 
                        int ierr = VT_funcdef(event_steal_name.c_str(), VT_NOCLASS, &event_steal);
                    VT_BEGIN_CONSTRAINED(event_steal);
                    #endif

                    stealTime += curtime() - start;
                    (*queues)[BCL::rank()].push(tId);

                    #if TRACE==1
                    VT_END_W_CONSTRAINED(event_steal);
                    #endif
                }

                j++;
            }

            // record the number of stolen tasks
            num_stolen_tasks[BCL::rank()] += j;

            long ownSize = (*queues)[BCL::rank()].size();
            if (ownSize > 0) {
                // printf("[R%ld] Successfully stolen %ld/%ld tasks!\n", BCL::rank(), 
                //                    (*queues)[BCL::rank()].size(), size);
                return true;
            }
        }
    }

    return false;
}


// ================================================================================
// Main Function
// ================================================================================

int main(int argc, char **argv)
{
    BCL::init(30 * 256, true);

    // initialize variables, get aguments and configs
    int tasks = 0;

    // resize the tracking vector of stolen tasks
    num_stolen_tasks.resize(BCL::nprocs(), 0);

    const char *tmp = getenv("STEALP");
    tmp = tmp ? tmp : "0.5";
    stealPerc = atof(tmp);

    tmp = getenv("STEALING");
    tmp = tmp ? tmp : "1";
    int stealing = atoi(tmp);

    // get the number of tasks to enqueue
    int rankOffset = 0;
    for (int i = 1; i < argc; i += 2) {
        if (atoi(argv[i + 1]) + rankOffset > BCL::rank() && rankOffset <= BCL::rank()) {
            tasks = atoi(argv[i]);
        }
        rankOffset += atoi(argv[i + 1]);
    }

    // executing program depeding on the STEALING env. variable
    double tsRuntime, noTsRuntime;
    if (stealing > 0)
        tsRuntime = execute(tasks, 1);

    if (stealing == 0 || stealing == 2)
        noTsRuntime = execute(tasks, 0);

    if (BCL::rank() == 0 && stealing == 2)
        printf("Speedup: %lf\n", noTsRuntime / tsRuntime);

    // finialize BCL
    BCL::finalize();

    return 0;
}
