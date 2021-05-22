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

bool verifyResults = false,
     ShowRuntimeDist = true;

double stealPerc, stealTime = 0, multiplyTime = 0, responseTime = 0;

void multiply(double *matrix, double *matrix2, double *result);
void initialize_matrix_rnd(double *mat);
bool steal(std::vector<BCL::CircularQueue<int>> *queues);

using namespace std;

static inline double curtime(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

double execute(int tasks, int taskStealing)
{
    double fTimeStart, fTimeEnd, elapsed = 0;

    if (BCL::rank() == 0)
        printf("Executing code with: RANKS:%ld  THREADS:%d  MSIZE:%d  STEALP:%lf  STEALING:%d\n", BCL::nprocs(), omp_get_max_threads(), MSIZE, stealPerc, taskStealing);

    //setting up global data structures
    std::vector<BCL::CircularQueue<int>> queues;
    std::vector<BCL::Array<BCL::GlobalPtr<task>>> taskPointerArray;
    for (size_t rank = 0; rank < BCL::nprocs(); rank++)
    {
        queues.push_back(BCL::CircularQueue<int>(rank, 30000));
        taskPointerArray.push_back(BCL::Array<BCL::GlobalPtr<task>>(rank, 1));
    }
    taskPointerArray[BCL::rank()][0].put(BCL::alloc<task>(tasks));

    //create and save tasks
    int nextId = 0;
    for (int i = 0; i < tasks; i++)
    {
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
    vector<BCL::GlobalPtr<task>> taskPtrs;
    for (size_t rank = 0; rank < BCL::nprocs(); rank++)
    {
        taskPtrs.push_back(taskPointerArray[rank][0].get());
    }

    //solve tasks and save the result at the origin rank
    fTimeStart = curtime();

    while (true)
    {
        double startMult = curtime();

#pragma omp parallel for
        for (int i = 0; i < queues[BCL::rank()].size(); i++)
        {
            // cout << BCL::rank() << " " << queues[BCL::rank()].size() <<"\n";
            int tId;
            bool success = queues[BCL::rank()].pop(tId);
            // cout << success << "\n";
            if (success)
            {
                task t;
                int ownerRank = tId >> 24;
                int taskIndex = (tId << 8) >> 8;

                // printf("[%ld, %ld]Processing task %d %d\n", BCL::rank(), omp_get_thread_num(), ownerRank, taskIndex);
                BCL::memcpy(&t, taskPtrs[ownerRank] + taskIndex, sizeof(task));

                multiply(t.matrix, t.matrix2, t.result);

                int offSet = 2 * MSIZE * MSIZE * sizeof(double) + sizeof(int);
                BCL::memcpy(taskPtrs[ownerRank] + taskIndex, (&t), sizeof(task));
            }
        }
        multiplyTime += curtime() - startMult;

        if (queues[BCL::rank()].size() == 0 && (!taskStealing || !steal(&queues)))
        {
            break;
        }
    }

    elapsed = curtime() - fTimeStart;
    if (ShowRuntimeDist)
        printf("[%ld]Steal Time: %lf%%  Multiply time: %lf%%  Response time: %lf%%\n",
               BCL::rank(), stealTime / elapsed * 100, multiplyTime / elapsed * 100, responseTime / elapsed * 100);

    // int flag;

    // printf("%d\n",MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE));
    BCL::barrier();

    if (BCL::rank() == 0)
    {
        fTimeEnd = curtime();
        elapsed = fTimeEnd - fTimeStart;
        printf("%lf\n", elapsed);
    }

    task t;
    if (verifyResults)
        for (int i = 0; i < tasks; i++)
        {
            t = taskPtrs[BCL::rank()][i];
            double *result = (double *)malloc(MSIZE * MSIZE * sizeof(double));
            multiply(t.matrix, t.matrix2, result);

            for (int i = 0; i < MSIZE * MSIZE; i++)
            {
                if (result[i] != t.result[i])
                {
                    printf("[%ld]The multiplication of task with TID: %d was computed incorrectly!\n", BCL::rank(), t.taskId);
                    break;
                }
            }
            free(result);
        }

    BCL::barrier();

    return elapsed;
}

void initialize_matrix_rnd(double *mat)
{
    double lower_bound = 0;
    double upper_bound = 10;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    for (int i = 0; i < MSIZE * MSIZE; i++)
    {
        mat[i] = unif(re);
    }
}

void multiply(double *matrix, double *matrix2, double *result)
{
    for (int i = 0; i < MSIZE * MSIZE; i += 1)
    {
        double value = 0;
        int k = i % MSIZE;
        for (int j = (i / MSIZE) * MSIZE; j < (i / MSIZE) * MSIZE + MSIZE; j++)
        {
            value = value + matrix[j] * matrix2[k];
            k += MSIZE;
        }
        result[i] = value;
    }
}

bool steal(std::vector<BCL::CircularQueue<int>> *queues)
{

    std::srand(unsigned(std::time(0)));
    std::vector<int> ranks;

    for (int i = 0; i < BCL::nprocs(); i++)
    {
        if (i != BCL::rank())
            ranks.push_back(i);
    }

    std::random_shuffle(ranks.begin(), ranks.end());

    //iterates through ranks and tests if they have tasks left
    for (std::vector<int>::iterator it = ranks.begin(); it != ranks.end(); ++it)
    {

        //measuring the time until the first response from foreign rank
        double start = curtime();
        long size = (*queues)[*it].size();
        responseTime += curtime() - start;

        int minAmount = omp_get_max_threads() - 1;
        if (size > 0)
        {
            //steals half the tasks
            int j = 0;
            while (j < (*queues)[*it].size() * stealPerc)
            {
                int tId;

                start = curtime();
                if ((*queues)[*it].pop(tId))
                {
                    stealTime += curtime() - start;
                    (*queues)[BCL::rank()].push(tId);
                }

                j++;
            }

            long ownSize = (*queues)[BCL::rank()].size();
            if (ownSize > 0)
            {
                // printf("[%ld]Successfully stolen %ld/%ld tasks!\n", BCL::rank(), (*queues)[BCL::rank()].size(), size);
                return true;
            }
        }
    }

    return false;
}

int main(int argc, char **argv)
{
    BCL::init(30 * 256, true);

    //initializing variables
    int tasks = 0;

    //obtaining env. variables
    const char *tmp = getenv("STEALP");
    tmp = tmp ? tmp : "0.5";
    stealPerc = atof(tmp);

    tmp = getenv("STEALING");
    tmp = tmp ? tmp : "1";
    int stealing = atoi(tmp);

    //get the number of tasks to enqueue
    int rankOffset = 0;
    for (int i = 1; i < argc; i += 2)
    {
        if (atoi(argv[i + 1]) + rankOffset > BCL::rank() && rankOffset <= BCL::rank())
        {
            tasks = atoi(argv[i]);
        }
        rankOffset += atoi(argv[i + 1]);
    }

    //executing program depeding on the STEALING env. variable
    double tsRuntime, noTsRuntime;
    if (stealing > 0)
        tsRuntime = execute(tasks, 1);

    if (stealing == 0 || stealing == 2)
        noTsRuntime = execute(tasks, 0);

    if (BCL::rank() == 0 && stealing == 2)
        printf("Speedup: %lf\n", noTsRuntime / tsRuntime);

    BCL::finalize();
}
