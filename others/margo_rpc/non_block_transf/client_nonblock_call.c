#include <assert.h>
#include <stdio.h>
#include <margo.h>
#include "inout_struct.h"

int main(int argc, char** argv)
{
    if(argc != 2) {
        fprintf(stderr,"Usage: %s <server address>\n", argv[0]);
        exit(0);
    }

    // init margo_client mode
    margo_instance_id mid = margo_init("tcp", MARGO_CLIENT_MODE, 0, 0);
    margo_set_log_level(mid, MARGO_LOG_DEBUG);
    hg_id_t sum_rpc_id = MARGO_REGISTER(mid, "sum", sum_in_t, sum_out_t, NULL);

    // look up for the addr of server
    hg_addr_t svr_addr;
    margo_addr_lookup(mid, argv[1], &svr_addr);

    // the same request, need to compute 4 sums
    // then, it calls the remote procedure at ser_side
    // four times
    int i;
    sum_in_t args;
    for(i = 0; i < 4; i++) {
        args.x = 42+i*2;
        args.y = 42+i*2+1;

        // Instead of using margo_forward, we use margo_iforward.
        // This function returns immediately after having sent the 
        // RPC to the server. It also takes an extra argument of
        // type margo_request*. The client will use this request
        // object to check the status of the RPC.
        hg_handle_t h;
        margo_create(mid, svr_addr, sum_rpc_id, &h);
        margo_request req;
        margo_iforward(h, &args, &req);

        // Note: It is safe to delete or modify the RPC’s input right
        // after the call to margo_iforward. margo_iforward indeed
        // returns after having serialized this input into its send buffer.

        margo_debug(mid, "Waiting for reply...");

        // We then use margo_wait on the request to block until we have
        // received a response from the server. Alternatively, margo_test
        // can be be used to check whether the server has sent a response,
        // without blocking if it hasn’t.
        margo_wait(req);

        sum_out_t resp;
        margo_get_output(h, &resp);

        margo_debug(mid, "Got response: %d+%d = %d", args.x, args.y, resp.ret);

        margo_free_output(h,&resp);
        margo_destroy(h);
    }

    margo_addr_free(mid, svr_addr);

    margo_finalize(mid);

    return 0;
}