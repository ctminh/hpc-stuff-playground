#include <assert.h>
#include <stdio.h>
#include <margo.h>
#include <sys/types.h>
#include <unistd.h>


int main(int argc, char *argv[]){
    // init margo, also create a margo instance id
    // init takes 4 args:
    //  + protocol (also work with ip_addr or port_num???)
    //  + MARGO_SER... specifies this is server_mode
    //  + the 3rd one indicates whether an Argobots execution stream (ES) should create or not
    //      if this argument is set to 0, the progress loop is going to run in the context of the
    //      main ES (this should be the standard scenario, unless you have a good reason for not
    //      using the main ES, such as the main ES using MPI primitives that could block the progress
    //      loop). A value of 1 will make Margo create an ES to run the Mercury progress loop. 
    //  + the 4th one is the number of ES to create and use for executing RPC handlers.
    //      A value of 0 will make Margo execute RPCs in the ES that called margo_init. A value of -1
    //      will make Margo execute the RPCs in the ES running the progress loop. A positive value
    //      will make Margo create new ESs to run the RPCs.
    margo_instance_id mid = margo_init("tcp", MARGO_SERVER_MODE, 0, -1);
    assert(mid);

    pid_t s_pid = getpid();
    printf("[SERVER] pid=%d: init margo OK...\n", s_pid);

    // get the address of the server, which is then converted into a string
    hg_addr_t my_address;
    margo_addr_self(mid, &my_address);
    char addr_str[128];
    size_t addr_str_size = 128;
    margo_addr_to_string(mid, addr_str, &addr_str_size, my_address);
    margo_addr_free(mid, my_address);

    // blocks the server in the Mercury progress loop until another ES calls
    margo_set_log_level(mid, MARGO_LOG_INFO);
    margo_info(mid, "[SERVER] running at address %s", addr_str);

    margo_wait_for_finalize(mid);
    margo_info(mid, "[SERVER] passed wait_for_finalize...");

    return 0;
}