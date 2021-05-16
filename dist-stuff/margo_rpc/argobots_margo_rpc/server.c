#include <assert.h>
#include <stdio.h>
#include <margo.h>
#include <provider_alpha_server.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

/**
 * Using the Alpha server.
 * 
 * A server can be written that spins up an Alpha providervas follows.
 * 
 * As mentioned, we saw that the alpha_provider_register function is taking
 * an ABT_pool argument that is passed down to MARGO_REGISTER_PROVIDER.
 * 
 * Argobots pools are a good way to assign resources (typically cores) to
 * particular providers. In the following example, we rewrite the server 
 * code in such a way that the Alpha provider gets its own execution stream.
 */


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
    margo_info(mid, "Server running at address %s, with provider id 42", addr_str);

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

    // After initializing Margo (which initializes Argobots), we create an Argobots
    // pool and an execution stream (ES) that will execute work (ULTs and tasklets)
    // from this pool.
    // 
    // We use ABT_POOL_ACCESS_MPSC as access type to indicate that there will be
    // multiple producers of work units (in particular, the ES running the Mercury
    // progress loop) and a single consumer of work units (the ES we are about to create).

    ABT_pool pool;
    ABT_pool_create_basic(
            ABT_POOL_FIFO,
            ABT_POOL_ACCESS_SPSC,
            ABT_TRUE,
            &pool);

    // ABT_xstream_create_basic is then used to create the ES. Because Margo is
    // initializing and finalizing Argobots, we need a way to destroy this ES before Margo
    // finalizes Argobots. Hence with use margo_push_finalize_callback to add a callback that
    // will be called upon finalizing Margo. This callback joins the ES and destroys it.
    ABT_xstream xstream;
    ABT_xstream_create_basic(
            ABT_SCHED_DEFAULT,
            1,
            &pool,
            ABT_SCHED_CONFIG_NULL,
            &xstream);

    // We pass the newly created pool to the alpha_provider_register function, which will
    // make the Alpha provider use this pool to execute its RPC handlers.
    alpha_provider_register(mid, 42, pool, ALPHA_PROVIDER_IGNORE);

    margo_wait_for_finalize(mid);

    return 0;
}

static void finalize_xstream_cb(void* data) {
    ABT_xstream xstream = (ABT_xstream)data;
    ABT_xstream_join(xstream);
    ABT_xstream_free(&xstream);
}