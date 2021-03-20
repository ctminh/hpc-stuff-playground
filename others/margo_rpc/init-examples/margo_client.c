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

    pid_t c_pid = getpid();
    printf("[CLIENT] pid=%d: init margo OK...\n", c_pid);

    // margo_finalize is used to finalize the margo_instance_id object.
    margo_finalize(mid);
    printf("[CLIENT] finalize margo_server OK!!!\n");

    return 0;
}