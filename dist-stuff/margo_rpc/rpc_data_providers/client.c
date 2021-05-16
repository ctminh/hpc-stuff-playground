#include <stdio.h>
#include <margo.h>
#include <provider_alpha_client.h>


/**
 * Using the Alpha client.
 * 
 * The previous codes can be compiled into two libraries, libalpha-client.{a,so}
 * and libalpha-server.{a,so}. The former will be used by client codes to use
 * the Alpha microservice as follows.
 * 
 * Tips: to avoid conflicts with other microservices, it is recommended to prefix
 * the name of the RPCs with the name of the service, as we did
 * here with “alpha_sum”.
 * 
 * Note: providers declaring RPCs with distinct names (i.e. providers from distinct
 * microservices) can have the same provider ids. The provider id is here to distinguish
 * providers of the same type within a given server.
 * 
 * Timeout: the margo_provider_forward_timed and margo_provider_iforward_timed can
 * be used when sending RPCs (in a blocking or non-blocking manner) to specify a
 * timeout in milliseconds after which the call (or result of margo_wait) will
 * be HG_TIMEOUT.
 */


int main(int argc, char** argv)
{
    if(argc != 3) {
        fprintf(stderr,"Usage: %s <server address> <provider id>\n", argv[0]);
        exit(0);
    }

    const char* svr_addr_str = argv[1];
    uint16_t    provider_id  = atoi(argv[2]);

    margo_instance_id mid = margo_init("tcp", MARGO_CLIENT_MODE, 0, 0);
    margo_set_log_level(mid, MARGO_LOG_INFO);

    hg_addr_t svr_addr;
    margo_addr_lookup(mid, svr_addr_str, &svr_addr);

    alpha_client_t alpha_clt;
    alpha_provider_handle_t alpha_ph;

    alpha_client_init(mid, &alpha_clt);

    alpha_provider_handle_create(alpha_clt, svr_addr, provider_id, &alpha_ph);

    int32_t result;
    alpha_compute_sum(alpha_ph, 45, 23, &result);

    alpha_provider_handle_release(alpha_ph);

    alpha_client_finalize(alpha_clt);

    margo_addr_free(mid, svr_addr);

    margo_finalize(mid);

    return 0;
}