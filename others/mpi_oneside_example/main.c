#include <stdio.h>
#include <time.h> 
#include <mpi.h>
#include <string.h>
#include <VT.h>

#define NUM_ELEMENT 10000
#define _tracing_enabled 1

#ifndef VT_BEGIN_CONSTRAINED
#define VT_BEGIN_CONSTRAINED(event_id) if (_tracing_enabled) VT_begin(event_id);
#endif

#ifndef VT_END_W_CONSTRAINED
#define VT_END_W_CONSTRAINED(event_id) if (_tracing_enabled) VT_end(event_id);
#endif

void test_itac(int call_id)
{
    // ------------------------ begin VT -----------------------------
    int event_testitac = -1;
    char event_name[12] = "test_itac";
    int itac_err = VT_funcdef(event_name, VT_NOCLASS, &event_testitac);
    VT_BEGIN_CONSTRAINED(event_testitac);
    // ---------------------------------------------------------------

    printf("Call_ID = %d\n", call_id);
    int i;
    int N = 10000;
    double result = 0.0;
    double pi = 3.14;
    for (i = 0; i < N; i++)
    {
        result += i * pi;
    }

    // ------------------------ end VT -------------------------------
    VT_END_W_CONSTRAINED(event_testitac);
    // ---------------------------------------------------------------
}

int main(int argc, char *argv[])
{
    int i, rank, num_procs, len;
    int localbuffer[NUM_ELEMENT];   // a local buffer on each rank
    int sharedbuffer[NUM_ELEMENT];  // a global-accessible buffer for all ranks
    char name[MPI_MAX_PROCESSOR_NAME];

    // buffers for p2p test
    int p2p_send_buffer[NUM_ELEMENT];
    int p2p_recv_buffer[NUM_ELEMENT];

    // measure time
    clock_t t_get_begin, t_get_end;
    clock_t t_put_begin, t_put_end;
    clock_t p2p_t_begin, p2p_t_end;

    // init MPI/rank
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Get_processor_name(name, &len);

    // create a windows/rank for global-access
    printf("R%d: create a shared-windows\n", rank);
    MPI_Win win;
    MPI_Win_create(sharedbuffer, NUM_ELEMENT, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // init values to the buffer
    for (i = 0; i < NUM_ELEMENT; i++){
        sharedbuffer[i] = rank * NUM_ELEMENT + i;
        localbuffer[i] = rank;

        p2p_send_buffer[i] = rank;
        p2p_recv_buffer[i] = 0;
    }

    // MPI point-to-point send & recv
    p2p_t_begin = clock();
    if (rank < num_procs - 1){
        // printf("[P2P_SEND] R%d: sending data to R%d\n", rank, rank+1);
        MPI_Send(&p2p_send_buffer[0], NUM_ELEMENT, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    }else{
        // printf("[P2P_SEND] R%d: sending data to R%d\n", rank, 0);
        MPI_Send(&p2p_send_buffer[0], NUM_ELEMENT, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    if (rank != 0){
        // printf("[P2P_RECV] R%d: receiving data from R%d\n", rank, rank-1);
        MPI_Recv(&p2p_recv_buffer[0], NUM_ELEMENT, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }else{
        // printf("[P2P_RECV] R%d: receiving data from R%d\n", rank, num_procs-1);
        MPI_Recv(&p2p_recv_buffer[0], NUM_ELEMENT, MPI_INT, num_procs-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    p2p_t_end = clock();

    // print measurement time
    double p2p_elapsed_time = ((double)(p2p_t_end - p2p_t_begin)) / CLOCKS_PER_SEC;
    if (rank < num_procs-1)
        printf("[P2P-ELAPSED-TIME] R%d-R%d send-recv = %f (s)\n", rank, rank+1, p2p_elapsed_time);
    else
        printf("[P2P-ELAPSED-TIME] R%d-R%d send-recv = %f (s)\n", rank, 0, p2p_elapsed_time);
    

    //////////////////////////////////////////////////////////////////////////
    ///////////////////////////// TEST MPI_PUT ///////////////////////////////
    MPI_Win_fence(0, win);
    /* For example: there are 4 ranks - R0, R1, R2, R3
        R0: puts data from local_buf to the shared_windows of R1
        R1: ...       ...       ...         ...         ...   R2
        R2: ...       ...       ...         ...         ...   R3
        R3: ...       ...       ...         ...         ...   R0
     */
    t_put_begin = clock();
    if (rank < num_procs-1){
        // printf("[PUT] R%d: putting data from loca_buf to shar_win at R%d\n", rank, rank+1);
        MPI_Put(&localbuffer[0], NUM_ELEMENT, MPI_INT, rank+1, 0, NUM_ELEMENT, MPI_INT, win);
    }
    else {
        // printf("[PUT] R%d: putting data from loca_buf to shar_win at R%d\n", rank, 0);
        MPI_Put(&localbuffer[0], NUM_ELEMENT, MPI_INT, 0, 0, NUM_ELEMENT, MPI_INT, win);
    }
    t_put_end = clock();

    MPI_Win_fence(0, win);

    // print the measurement for PUT
    double put_elapsed_time = ((double)(t_put_end - t_put_begin)) / CLOCKS_PER_SEC; // in seconds
    if (rank < num_procs-1)
        printf("[ELAPSED-TIME] R%d - PUT to R%d = %f (s)\n", rank, rank+1, put_elapsed_time);
    else
        printf("[ELAPSED-TIME] R%d - PUT to R%d = %f (s)\n", rank, 0, put_elapsed_time);

    //////////////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////////////////////
    ///////////////////////////// TEST MPI_GET ///////////////////////////////
    /* Fence is one of the synchronization models used in active target communication.
    The MPI_Win_fenceroutine synchronizes RMA operations on a specified window. It is a
    collective call over the processgroup of the window. The fence is like a barrier: it synchronizes
    a sequence of RMA calls (e.g. put, get,accumulate) and it should be used before and after that sequence */
    MPI_Win_fence(0, win);

    t_get_begin = clock();
    if (rank != 0){
        // printf("[GET] R%d: getting data from shar_win at R%d\n", rank, rank-1);
        MPI_Get(&localbuffer[0], NUM_ELEMENT, MPI_INT, rank-1, 0, NUM_ELEMENT, MPI_INT, win);
    }
    else{
        // printf("[GET] R%d: getting data from shar_win at R%d\n", rank, num_procs-1);
        MPI_Get(&localbuffer[0], NUM_ELEMENT, MPI_INT, num_procs-1, 0, NUM_ELEMENT, MPI_INT, win);
    }
    t_get_end = clock();

    // set synchronization
    MPI_Win_fence(0, win);

    // print the measurement for GET
    double get_elapsed_time = ((double)(t_get_end - t_get_begin)) / CLOCKS_PER_SEC; // in seconds
    if (rank != 0)
        printf("[ELAPSED-TIME] R%d - GET from R%d = %f (s)\n", rank, rank-1, get_elapsed_time);
    else
        printf("[ELAPSED-TIME] R%d - GET from R%d = %f (s)\n", rank, num_procs-1, get_elapsed_time);

    //////////////////////////////////////////////////////////////////////////

    // set synchronization
    MPI_Win_fence(0, win);

    // check 10-elems of the result buff at R0 / R1
    if (rank == 0){
        printf("[CHECK] R%d: recv-buffer (P2P) and local-buffer (GET)\n", rank);
        for (i = 0; i < 10; i++){
            printf("%d ", p2p_recv_buffer[i]);
        } printf("\n");

        for (i = 0; i < 10; i++){
            printf("%d ", localbuffer[i]);
        } printf("\n");
    }

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}