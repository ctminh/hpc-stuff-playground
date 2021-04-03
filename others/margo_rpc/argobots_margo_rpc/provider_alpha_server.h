#ifndef __ALPHA_SERVER_H
#define __ALPHA_SERVER_H

#include <margo.h>
#include "provider_alpha_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ALPHA_ABT_POOL_DEFAULT ABT_POOL_NULL

/**
 * The alpha_provider_t handle.
 *
 * This interface contains the definition of an opaque pointer type,
 * alpha_provider_t, which will be used to hide the implementation
 * of our Alpha provider.
 *
 */
typedef struct alpha_provider* alpha_provider_t;
#define ALPHA_PROVIDER_NULL ((alpha_provider_t)NULL)
#define ALPHA_PROVIDER_IGNORE ((alpha_provider_t*)NULL)

/**
 * @brief Creates a new ALPHA provider. If ALPHA_PROVIDER_IGNORE
 * is passed as last argument, the provider will be automatically
 * destroyed when calling :code:`margo_finalize`.
 *
 * @param[in] mid Margo instance
 * @param[in] provider_id provider id
 * @param[in] pool Argobots pool
 * @param[out] provider provider handle
 *
 * @return ALPHA_SUCCESS or error code defined in alpha-common.h
 */
int alpha_provider_register(
        margo_instance_id mid,
        uint16_t provider_id,
        ABT_pool pool,
        alpha_provider_t* provider);

/**
 * Note: The alpha_provider_register function also takes an Argobots pool
 * as argument. We will discuss this in a following tutorial.
 * 
 * This interface would also be the place where to put other
 * functions that configure or modify the Alpha provider once created.
 */


/**
 * @brief Destroys the Alpha provider and deregisters its RPC.
 *
 * @param[in] provider Alpha provider
 *
 * @return ALPHA_SUCCESS or error code defined in alpha-common.h
 */
int alpha_provider_destroy(
        alpha_provider_t provider);

#ifdef __cplusplus
}
#endif

#endif