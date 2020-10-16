#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <VT.h>

#define NUM_ELEMENT 4

void test_itac(int call_id)
{
    printf("Call_ID = %d\n", call_id);

    // call VT
    int event_testitac = -1;
    char event_name[12] = "test_itac";
    int itac_err = VT_funcdef(event_name, VT_NOCLASS, &event_testitac);
    VT_BEGIN_CONSTRAINED(event_testitac);
}

int main(int argc, char *argv[])
{
    int i, id, num_procs, len, localbuffer[NUM_ELEMENT], sharedbuffer[NUM_ELEMENT];
    char name[MPI_MAX_PROCESSOR_NAME];

    MPI_Win win;    // called by all processes to create a window of shared_mem
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Get_processor_name(name, &len);

    printf("Rank %d running on %s\n", id, name);
    for (i = 0; i < 10; i++){
        test_itac(i);
    }

    MPI_Win_create(sharedbuffer, NUM_ELEMENT, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // add values to the buffer
    for (i = 0; i < NUM_ELEMENT; i++)
    {
        sharedbuffer[i] = 10*id + i;
        localbuffer[i] = 0;
    }

    printf("Rank %d sets data in the shared memory:\n", id);
    for (i = 0; i < NUM_ELEMENT; i++)
        printf(" %02d", sharedbuffer[i]);
    printf("\n-------------------------------------------\n");

    ////////////////// open MPI_Win_fence for getting data from remote process ///////////////////
    MPI_Win_fence(0, win);

    // MPI get data from local buffers
    if (id != 0)
        MPI_Get(&localbuffer[0], NUM_ELEMENT, MPI_INT, id-1, 0, NUM_ELEMENT, MPI_INT, win);
    else
        MPI_Get(&localbuffer[0], NUM_ELEMENT, MPI_INT, num_procs-1, 0, NUM_ELEMENT, MPI_INT, win);

    MPI_Win_fence(0, win);
    /////////////////////////////////////////////////////////////////////////////////////////////

    printf("Rank %d gets data from the shared memory:\n", id);
    for (i = 0; i < NUM_ELEMENT; i++)
        printf(" %02d", localbuffer[i]);
    printf("\n-------------------------------------------\n");


    ////////////////// open MPI_Win_fence for putting data to remote process ///////////////////
    MPI_Win_fence(0, win);

    if (id < num_procs-1)
        MPI_Put(&localbuffer[0], NUM_ELEMENT, MPI_INT, id+1, 0, NUM_ELEMENT, MPI_INT, win);
    else
        MPI_Put(&localbuffer[0], NUM_ELEMENT, MPI_INT, 0, 0, NUM_ELEMENT, MPI_INT, win);

    MPI_Win_fence(0, win);

    printf("Rank %d has new data in the shared memory:\n", id);
    for (i = 0; i < NUM_ELEMENT; i++)
        printf(" %02d-at-%p", sharedbuffer[i], &sharedbuffer[i]);
    printf("\n-------------------------------------------\n");
    

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}