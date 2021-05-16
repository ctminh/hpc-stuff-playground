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

    alpha_provider_register(mid, 42, ALPHA_ABT_POOL_DEFAULT, ALPHA_PROVIDER_IGNORE);

    margo_wait_for_finalize(mid);

    return 0;
}