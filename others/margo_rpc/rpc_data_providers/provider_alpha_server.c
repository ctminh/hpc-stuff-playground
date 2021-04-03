#include "provider_alpha_server.h"
#include "provider_inout_struct.h"


/**
 * Defining the alpha_provider structure.
 * 
 * It may contain the RPC ids as well as any data you may need as
 * context for your RPCs.
 */
struct alpha_provider {
    margo_instance_id mid;
    hg_id_t sum_id;
    /* other provider-specific data */
};

static void alpha_finalize_provider(void* p);

DECLARE_MARGO_RPC_HANDLER(alpha_sum_ult);
static void alpha_sum_ult(hg_handle_t h);
/* add other RPC declarations here */

int alpha_provider_register(
        margo_instance_id mid,
        uint16_t provider_id,
        ABT_pool pool,
        alpha_provider_t* provider)
{
    alpha_provider_t p;
    hg_id_t id;
    hg_bool_t flag;

    // The alpha_provider_register function starts by checking that the Margo
    // instance is in server mode by using margo_is_listening. It then checks
    // that there isn’t already an alpha provider with the same id.
    flag = margo_is_listening(mid);
    if(flag == HG_FALSE) {
        margo_error(mid, "alpha_provider_register(): margo instance is not a server");
        return ALPHA_FAILURE;
    }

    // It does so by using margo_provider_registered_name to check whether the
    // sum RPC has already been registered with the same provider id.
    margo_provider_registered_name(mid, "alpha_sum", provider_id, &id, &flag);
    if(flag == HG_TRUE) {
        margo_error(mid, "alpha_provider_register(): a provider with the same provider id (%d) already exists", provider_id);
        return ALPHA_FAILURE;
    }

    p = (alpha_provider_t)calloc(1, sizeof(*p));
    if(p == NULL) {
        margo_error(mid, "alpha_provider_register(): failed to allocate memory for provider");
        return ALPHA_FAILURE;
    }

    p->mid = mid;

    // We then use MARGO_REGISTER_PROVIDER instead of MARGO_REGISTER. This macro
    // takes a provider id and an Argobots pool in addition to the parameters of
    // MARGO_REGISTER.
    id = MARGO_REGISTER_PROVIDER(mid, "alpha_sum",
            sum_in_t, sum_out_t,
            alpha_sum_ult, provider_id, pool);
    margo_register_data(mid, id, (void*)p, NULL);
    p->sum_id = id;
    /* add other RPC registration here */

    // Finally, we call margo_provider_push_finalize_callback to setup a callback
    // that Margo should call when calling margo_finalize. This callback will
    // deregister the RPCs and free the provider.
    margo_provider_push_finalize_callback(mid, p, &alpha_finalize_provider, p);

    if(provider)
        *provider = p;
    return ALPHA_SUCCESS;
}

static void alpha_finalize_provider(void* p)
{
    alpha_provider_t provider = (alpha_provider_t)p;
    margo_deregister(provider->mid, provider->sum_id);
    /* deregister other RPC ids ... */
    free(provider);
}

int alpha_provider_destroy(
        alpha_provider_t provider)
{
    // In most cases the user will create a provider and leave it running until
    // something calls margo_finalize, at which point the provider’s 
    // inalization callback will be called.

    // If the user wants to destroy the provider before Margo is finalized, it is
    // important to tell Margo not to call the provider’s finalization callback
    // when margo_finalize. Hence, we use margo_provider_pop_finalize_callback.
    //
    // This function takes a Margo instance, and an owner for the callback
    // (here the provider). If the provider registered multiple callbacks using
    // margo_provider_push_finalize_callback,
    // margo_provider_pop_finalize_callback will pop the last one pushed,
    // and should therefore be called as many time as needed to pop all the
    // finalization callbacks corresponding to the provider.

    /* pop the finalize callback */
    margo_provider_pop_finalize_callback(provider->mid, provider);

    // Warning: Finalization callbacks are called after the Mercury progress loop is terminated.
    // Hence, you cannot send RPCs from them. If you need a finalization callback to
    // be called before the progress loop is terminated, use margo_push_prefinalize_callback
    // or margo_provider_push_prefinalize_callback.

    /* call the callback */
    alpha_finalize_provider(provider);

    return ALPHA_SUCCESS;
}


static void alpha_sum_ult(hg_handle_t h)
{
    hg_return_t ret;
    sum_in_t     in;
    sum_out_t   out;

    margo_instance_id mid = margo_hg_handle_get_instance(h);

    const struct hg_info* info = margo_get_info(h);
    alpha_provider_t provider = (alpha_provider_t)margo_registered_data(mid, info->id);

    ret = margo_get_input(h, &in);

    out.ret = in.x + in.y;
    margo_trace(mid, "Computed %d + %d = %d", in.x, in.y, out.ret);

    ret = margo_respond(h, &out);
    ret = margo_free_input(h, &in);
    margo_destroy(h);
}
DEFINE_MARGO_RPC_HANDLER(alpha_sum_ult)