#include <assert.h>
#include <stdio.h>
#include <margo.h>
#include "inout_struct.h"

/**
 * Main Function
 */
int main(int argc, char** argv)
{
    if(argc != 2) {
        fprintf(stderr,"Usage: %s <server address>\n", argv[0]);
        exit(0);
    }

    margo_instance_id mid = margo_init("tcp", MARGO_CLIENT_MODE, 0, 0);
    margo_set_log_level(mid, MARGO_LOG_INFO);

    // Initialize the sum_in_t args; and sum_out_t resp; to hold respectively
    // the arguments of the RPC (what will become the in variable on the server
    // side) and the return value (out on the server side).
    hg_id_t sum_rpc_id = MARGO_REGISTER(mid, "sum", sum_in_t, sum_out_t, NULL);

    hg_addr_t svr_addr;
    margo_addr_lookup(mid, argv[1], &svr_addr);

    // compute 4 times, means we call the sum func at the server_side 4 times.
    int i;
    sum_in_t args;
    for(i = 0; i < 4; i++) {
        args.x = 42+i*2;
        args.y = 42+i*2+1;

        // margo_forward now takes a pointer to the input argument
        // as second parameter
        hg_handle_t h;
        margo_create(mid, svr_addr, sum_rpc_id, &h);
        margo_forward(h, &args);

        // and margo_get_output is used to deserialized the value
        // returned by the server into the resp variable.
        sum_out_t resp;
        margo_get_output(h, &resp);

        margo_info(mid, "Got response: %d+%d = %d\n", args.x, args.y, resp.ret);

        // Just like we called margo_free_input on the server because the
        // input had been obtained using margo_get_input, we must call
        // margo_free_output on the client side because the output has been
        // obtained using margo_get_output
        margo_free_output(h,&resp);
        margo_destroy(h);
    }

    /**
    * Timeout
    * 
    * It can sometimes be important for the client to be able to timeout if an operation
    * takes too long. This can be done using margo_forward_timed, which takes an extra
    * parameter: a timeout (double) value in milliseconds. If the server has not responded
    * to the RPC after this timeout expires, margo_forward_timed will return HG_TIMEOUT and
    * the RPC will be cancelled.
    */

    margo_addr_free(mid, svr_addr);

    margo_finalize(mid);

    return 0;
}