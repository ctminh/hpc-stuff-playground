#include <assert.h>
#include <stdio.h>
#include <margo.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    /* This client takes the server’s address as argument (copy-past the
    address printed by the server when calling the client). This string
    representation of the server’s address must be resolved into a hg_addr_t
    object. This is done by margo_addr_lookup. */
    if(argc != 2) {
        fprintf(stderr,"Usage: %s <server address>\n", argv[0]);
        exit(0);
    }else
        printf("[CLIENT] Got the server_addr - %s...\n", argv[1]);

    // init margo-client
    margo_instance_id mid = MARGO_INSTANCE_NULL;
    mid = margo_init("tcp", MARGO_CLIENT_MODE, 0, 0);
    assert(mid);

    // get the address of the client, which is then converted into a string
    hg_addr_t my_address;
    margo_addr_self(mid, &my_address);
    char addr_str[128];
    size_t addr_str_size = 128;
    margo_addr_to_string(mid, addr_str, &addr_str_size, my_address);
    margo_addr_free(mid, my_address);

    // get process id of the client
    pid_t c_pid = getpid();
    printf("[CLIENT] pid=%d: init margo OK at %s...\n", c_pid, addr_str);

    // register the function to be called at the server side
    hg_id_t hello_rpc_id = MARGO_REGISTER(mid, "hello", void, void, NULL);
    margo_registered_disable_response(mid, hello_rpc_id, HG_TRUE);

    // Once resolved, the address can be used in a call to margo_create to create a
    // hg_handle_t object. The hg_handle_t object represents an RPC request ready to
    // be sent to the server.
    hg_return_t ret;
    hg_addr_t svr_addr;
    ret = margo_addr_lookup(mid, argv[1], &svr_addr);
    assert(ret == HG_SUCCESS);

    hg_handle_t handle;
    ret = margo_create(mid, svr_addr, hello_rpc_id, &handle);
    assert(ret == HG_SUCCESS);

    // margo_forward effectively sends the request to the server.
    // We pass NULL as a second argument because the RPC does not take any input.
    ret = margo_forward(handle, NULL);
    assert(ret == HG_SUCCESS);

    /* Because we have called margo_registered_disable_response, Margo knows that
    the client should not expect a response from the server, hence margo_forward will
    return immediately. We then destroy the handle using margo_destroy, free the hg_addr_t
    object using margo_addr_free, and finalize Margo. */
    ret = margo_destroy(handle);
    assert(ret == HG_SUCCESS);

    ret = margo_addr_free(mid, svr_addr);
    assert(ret == HG_SUCCESS);

    margo_finalize(mid);

    return 0;
}