#include <assert.h>
#include <stdio.h>
#include <margo.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char** argv)
{
    // call margo_init with MARGO_CLIENT_MODE to indicate that this is a client.
    // just like the server with the args
    margo_instance_id mid = margo_init("tcp", MARGO_CLIENT_MODE, 0, 0);
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
    printf("[CLIENT] pid=%d: init margo OK at %s...\n", c_pid, my_address);

    // margo_finalize is used to finalize the margo_instance_id object.
    margo_finalize(mid);
    printf("[CLIENT] finalize margo_server OK!!!\n");

    return 0;
}