#include <assert.h>
#include <stdio.h>
#include <margo.h>
#include "inout_struct.h"
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

static const int TOTAL_RPCS = 16;
static int num_rpcs = 0;

static void sum(hg_handle_t h);
DECLARE_MARGO_RPC_HANDLER(sum)

int main(int argc, char** argv)
{
    margo_instance_id mid = margo_init("tcp", MARGO_SERVER_MODE, 0, 0);
    assert(mid);
    margo_set_log_level(mid, MARGO_LOG_INFO);

    hg_addr_t my_address;
    margo_addr_self(mid, &my_address);
    char addr_str[128];
    size_t addr_str_size = 128;
    margo_addr_to_string(mid, addr_str, &addr_str_size, my_address);
    margo_addr_free(mid,my_address);

    margo_info(mid, "Server running at address %s\n", addr_str);

    // write the addr to file, then scriptor'd catch it
    // and pass it to client-side
    FILE *file_server_addr;
    file_server_addr = fopen("./f_server_addr.txt", "w");
    if(file_server_addr == NULL){
        printf("Error: cannot open file to write!");   
        exit(1);             
    }
    fprintf(file_server_addr, "%s", addr_str);
    fclose(file_server_addr);

    MARGO_REGISTER(mid, "sum", sum_in_t, sum_out_t, sum);

    margo_wait_for_finalize(mid);

    return 0;
}

static void sum(hg_handle_t h)
{
    hg_return_t ret;
    num_rpcs += 1;

    sum_in_t in;
    sum_out_t out;
    int32_t* values;
    hg_bulk_t local_bulk;

    margo_instance_id mid = margo_hg_handle_get_instance(h);

    const struct hg_info* info = margo_get_info(h);
    hg_addr_t client_addr = info->addr;

    ret = margo_get_input(h, &in);
    assert(ret == HG_SUCCESS);

    // Within the RPC handler, after deserializing the RPC’s input,
    // we allocate an array of appropriate size
    values = calloc(in.n, sizeof(*values));
    hg_size_t buf_size = in.n * sizeof(*values);

    // expose it the same way as we did on the client side, to get
    // a local bulk handle, using margo_bulk_create. This time we 
    // specify that this handle will be only written.
    ret = margo_bulk_create(mid, 1, (void**)&values, &buf_size,
            HG_BULK_WRITE_ONLY, &local_bulk);
    assert(ret == HG_SUCCESS);

    // margo_bulk_transfer is used to do the transfer. Here we pull
    // (HG_BULK_PULL) the data from the client’s memory to the server’s
    // local memory. We provide the client’s address (obtained from the
    // hg_info structure of the RPC handle), the offset in the client’s memory
    // region (here 0) and on the local memory region (0 as well),
    // as well as the size in bytes.
    ret = margo_bulk_transfer(mid, HG_BULK_PULL, client_addr,
            in.bulk, 0, local_bulk, 0, buf_size);
    assert(ret == HG_SUCCESS);

    // Once the transfer is completed, we perform the sum,
    out.ret = 0;
    int i;
    for(i = 0; i < in.n; i++) {
        out.ret += values[i];
    }

    // and return it to the client.
    ret = margo_respond(h, &out);
    assert(ret == HG_SUCCESS);

    // We don’t forget to use margo_bulk_free to free the bulk handle we created
    // (the bulk handle in the in structure will be freed by margo_free_input,
    // which is why it is so important that this function be called).
    ret = margo_bulk_free(local_bulk);
    assert(ret == HG_SUCCESS);

    free(values);

    ret = margo_free_input(h, &in);
    assert(ret == HG_SUCCESS);

    ret = margo_destroy(h);
    assert(ret == HG_SUCCESS);

    if(num_rpcs == TOTAL_RPCS) {
        margo_finalize(mid);
    }
}
DEFINE_MARGO_RPC_HANDLER(sum)