#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>
#include <bcl/containers/CircularQueue.hpp>
#include <string>
#include <cstdlib>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <bitset>

#include <random>

bool verifyResults = false;
double stealPerc, stealTime = 0;
int taskSize;
void multiply(double *matrix, double *matrix2, double *result, int matrixSize);
void initialize_matrix_rnd(double *mat, int matrixSize);
bool steal(std::vector<BCL::CircularQueue<task>> *queues);

static inline double curtime(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

double execute(int tasks, int matrixSize, int taskStealing)
{
    double fTimeStart, fTimeEnd, elapsed = 0;

    if (BCL::rank() == 0)
        printf("Executing code with: RANKS:%ld  THREADS:%d  MSIZE:%d  STEALP:%lf  STEALING:%d\n", BCL::nprocs(), omp_get_max_threads(), matrixSize, stealPerc, taskStealing);

    std::vector<BCL::CircularQueue<task>> queues;
    std::vector<BCL::Array<task>> finishedTasks;
    for (size_t rank = 0; rank < BCL::nprocs(); rank++)
    {
        queues.push_back(BCL::CircularQueue<task>(rank, 30000));
        finishedTasks.push_back(BCL::Array<task>(rank, 3000));
    }


    int nextId = 0;
    //create tasks
    for (int i = 0; i < tasks; i++)
    {
        task t;
        t.matrixSize = matrixSize;
        t.taskId = (int)(BCL::rank());

        t.taskId <<= 24;
        t.taskId += nextId;
        nextId++;
        initialize_matrix_rnd(t.matrix, matrixSize);
        initialize_matrix_rnd(t.matrix2, matrixSize);
        queues[BCL::rank()].push(t, BCL::CircularQueueAL::push);
    }
    BCL::barrier();

    //solve tasks
    if (BCL::rank() == 0)
    {
        fTimeStart = curtime();
    }

    while (true)
    {
#pragma omp parallel
        {
#pragma omp for
            for (int i = 0; i < queues[BCL::rank()].size(); i++)
            {

                task t;

                bool success = queues[BCL::rank()].pop(t);
                if (success)
                {
                    multiply(t.matrix, t.matrix2, t.result, matrixSize);
                    int rank = (t.taskId >> 24);
                    // std::string binary = std::bitset<32>(t.taskId).to_string(); //to binary
                    // std::cout<<binary<<"\n";
                    // printf("[%ld]Origin Rank %d\n", BCL::rank(), rank);
                    int adress = (t.taskId << 8)>>8;
                    finishedTasks[rank].put(adress, t);
                    
                }
            }
        }
        
        if (!taskStealing || !steal(&queues)){
            break;
        }
    }

    BCL::barrier();
    // for (int i = 0; i < BCL::nprocs(); i++)
    //     printf("[%ld] Rank %d:%ld\n", BCL::rank(), i, queues[i].size());

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
            t = finishedTasks[BCL::rank()][i];
            double *result = (double *)malloc(matrixSize * matrixSize * sizeof(double));
            multiply(t.matrix, t.matrix2, result, matrixSize);

            for (int i = 0; i < matrixSize * matrixSize; i++)
            {
                if (result[i] != t.result[i])
                    printf("[%ld]The multiplication was computed incorrectly!\n", BCL::rank());
            }
            free(result);
        }

    BCL::barrier();

    return elapsed;
}

void initialize_matrix_rnd(double *mat, int matrixSize)
{
    double lower_bound = 0;
    double upper_bound = 10;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        mat[i] = unif(re);
    }
}

void multiply(double *matrix, double *matrix2, double *result, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i += 1)
    {
        double value = 0;
        int k = i % matrixSize;
        for (int j = (i / matrixSize) * matrixSize; j < (i / matrixSize) * matrixSize + matrixSize; j++)
        {
            value = value + matrix[j] * matrix2[k];
            k += matrixSize;
        }
        result[i] = value;
    }
}

bool steal(std::vector<BCL::CircularQueue<task>> *queues)
{
    std::srand(unsigned(std::time(0)));
    std::vector<int> ranks;

    for (int i = 0; i < BCL::nprocs(); i++)
    {
        if (i != BCL::rank())
            ranks.push_back(i);
    }

    std::random_shuffle(ranks.begin(), ranks.end());

    int i = (BCL::rank() + 1) % BCL::nprocs();
    //iterates through ranks and tests if they have tasks left
    for (std::vector<int>::iterator it = ranks.begin(); it != ranks.end(); ++it)
    {
        // printf("[%ld]Current: %d\n", BCL::rank(), *it);
        //  printf("[%ld]Stealing from %d\n", BCL::rank(), *it);
        long size = (*queues)[*it].size();
        int minAmount = omp_get_max_threads() - 1;
        if (size > std::max(minAmount, 10))
        // if (size > 1)
        {
            //printf("[%ld]found %d!\n", BCL::rank(),i);
            task t;

            //steals half the tasks
            int j = 0;
            while (j < (*queues)[*it].size() * stealPerc)
            {
                if ((*queues)[*it].pop(t))
                    (*queues)[BCL::rank() * omp_get_num_threads() + omp_get_thread_num()].push(t);

                j++;
            }
            long ownSize = (*queues)[BCL::rank()].size();
            if (ownSize > 0)
            {
                //  printf("[%ld]Successfully stolen %ld/%ld tasks!\n", BCL::rank(), (*queues)[BCL::rank()].size(), size);
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
    int matrixSize = atoi(argv[1]);

    //obtaining env. variables
    const char *tmp = getenv("STEALP");
    tmp = tmp ? tmp : "0.5";
    stealPerc = atof(tmp);

    tmp = getenv("STEALING");
    tmp = tmp ? tmp : "1";
    int stealing = atoi(tmp);

    //get the number of tasks to enqueue
    int rankOffset = 0;
    for (int i = 2; i < argc; i += 2)
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
        tsRuntime = execute(tasks, matrixSize, 1);

    if (stealing == 0 || stealing == 2)
        noTsRuntime = execute(tasks, matrixSize, 0);

    if (BCL::rank() == 0 && stealing == 2)
        printf("Speedup: %lf\n", noTsRuntime / tsRuntime);

    BCL::finalize();
}
