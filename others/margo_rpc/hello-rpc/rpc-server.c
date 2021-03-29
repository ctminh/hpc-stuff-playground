#include <assert.h>
#include <stdio.h>
#include <margo.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

// declare num total rpc requests
static const int TOTAL_RPCS = 4;
static int num_rpcs = 0;

// declare the function would be called
// DECLARE_MARGO_RPC_HANDLER is a macro that generates the code
// necessary for the RPC handler to be placed in an Argobots user-level thread.
static void hello_world(hg_handle_t h);
DECLARE_MARGO_RPC_HANDLER(hello_world)

int main(int argc, char** argv)
{
    margo_instance_id mid = margo_init("tcp", MARGO_SERVER_MODE, 0, -1);
    assert(mid);

    hg_addr_t my_address;
    margo_addr_self(mid, &my_address);
    char addr_str[128];
    size_t addr_str_size = 128;
    margo_addr_to_string(mid, addr_str, &addr_str_size, my_address);
    margo_addr_free(mid,my_address);

    margo_set_log_level(mid, MARGO_LOG_INFO);
    pid_t s_pid = getpid();
    margo_info(mid, "[SERVER] pid=%d is running at address %s", s_pid, addr_str);

    // write the addr to file, then scriptor'd catch it and pass it to client-side
    FILE *file_server_addr;
    file_server_addr = fopen("./f_server_addr.txt", "w");
    if(file_server_addr == NULL){
        printf("Error: cannot open file to write!");   
        exit(1);             
    }
    fprintf(file_server_addr, "%s", addr_str);
    fclose(file_server_addr);

    // The two lines that register the RPC handler in the Margo instance are the following
    //  + MARGO_REGISTER is a macro that registers the RPC handler. Its first argument is the
    //    Margo instance. The second is the name of the RPC. The third and fourth are the
    //    types of the RPC’s input and output, respectively. The last parameter is the function
    //    we want to use as RPC handler
    //  + margo_registered_disable_response is used to indicate that this RPC handler does not
    //    send a response back to the client.
    hg_id_t rpc_id = MARGO_REGISTER(mid, "hello", void, void, hello_world);
    margo_registered_disable_response(mid, rpc_id, HG_TRUE);

    margo_wait_for_finalize(mid);

    return 0;
}

/* The rest of the program defines the hello_world function.
From inside an RPC handler, we can access the Margo instance using margo_hg_handle_get_instance.
This is the prefered method for better code organization, rather than declaring the Margo instance
as a global variable. The RPC handler must call margo_destroy on the hg_handle_t argument it is
being passed, after we are done using it.
    - The question is: how to pass h??? Maybe pass it at the client side...
*/
static void hello_world(hg_handle_t h)
{
    hg_return_t ret;

    margo_instance_id mid = margo_hg_handle_get_instance(h);

    margo_info(mid, "Hello World!");
    num_rpcs += 1;

    ret = margo_destroy(h);
    assert(ret == HG_SUCCESS);

    if(num_rpcs == TOTAL_RPCS) {
        margo_finalize(mid);
    }
}

DEFINE_MARGO_RPC_HANDLER(hello_world)