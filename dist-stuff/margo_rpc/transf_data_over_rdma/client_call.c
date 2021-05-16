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

    margo_instance_id mid = margo_init("tcp", MARGO_CLIENT_MODE, 0, 0);
    margo_set_log_level(mid, MARGO_LOG_DEBUG);

    hg_id_t sum_rpc_id = MARGO_REGISTER(mid, "sum", sum_in_t, sum_out_t, NULL);

    hg_addr_t svr_addr;
    margo_addr_lookup(mid, argv[1], &svr_addr);

    int i;
    sum_in_t args;
    for(i=0; i<4; i++) {

        // allocate the values buffer as an array of 10 integers (this array is on
        // the stack in this example. An array allocated on the heap would work
        // just the same.)
        int32_t values[10] = { 1,4,2,5,6,3,5,3,2,5 };
        hg_size_t segment_sizes[1] = { 10*sizeof(int32_t) };
        void* segment_ptrs[1] = { (void*)values };

        // margo_bulk_create is used to create an hg_bulk_t handle representing
        // the segment of memory exposed by the client.
        //      + 1st param: is the margo_instance_id.
        //      + 2nd param: the number of segments to expose
        //      + 3rd param: a void** array of addresses pointing to each segment
        //      + 4th param: a hg_size_t* array of sizes for each segment,
        //      + 5th param: the mode used to expose the memory region (here is read-only)
        //                   i.e., the server will only pull from this segment.
        //      + 6th param: an hg_bulk_t handle epresenting the segment of memory exposed by the client
        hg_bulk_t local_bulk;
        margo_bulk_create(mid, 1, segment_ptrs, segment_sizes, HG_BULK_READ_ONLY, &local_bulk);

        args.n = 10;
        args.bulk = local_bulk;

        hg_handle_t h;
        margo_create(mid, svr_addr, sum_rpc_id, &h);
        margo_forward(h, &args);

        sum_out_t resp;
        margo_get_output(h, &resp);

        margo_debug(mid, "Got response: %d", resp.ret);

        margo_free_output(h,&resp);
        margo_destroy(h);

        // the bulk handle is freed after being used,
        margo_bulk_free(local_bulk);
    }

    margo_addr_free(mid, svr_addr);

    margo_finalize(mid);

    return 0;
}