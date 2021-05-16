#ifndef __TYPES_H
#define __TYPES_H

#include <mercury.h>

/**
 * Serializing/deserializing data structures.
 * 
 * In previous tutorials, we have always used structures that can be
 * defined using Mercury’s MERCURY_GEN_PROC macro. If the structure
 * contains pointers, things get more complicated.
 */


/**
 * Let’s assume that we have a type int_list_t that represents a
 * pointer to a linked list of integers.
 */
typedef struct int_list {
    int32_t          value;
    struct int_list* next;
} *int_list_t;

/**
 * Init the int_list
 * 
 * More generally for any custom type X that we want to send or receive,
 * and that hasn’t been created using the Mercury macro, we need a
 * function of the form
 * 
 * @param hg_return_t hg_proc_X(hg_proc_t proc, void *data)
 * 
 * Any proc function must have three part, separated by a switch.
 * 
 * Note that here the type we are processing is int_list_t, so the
 * void* data argument is actually a pointer to an int_list_t, which is
 * itself a pointer to a structure.
 * 
 * We use the hg_proc_int32_t and hg_proc_hg_size_t functions to
 * serialize/deserialize int32_t and hg_size_t respectively. Most basic
 * datatypes have such a function defined in Mercury. To serialize or
 * deserialize raw memory, you can use
 * hg_proc_raw(hg_proc_t proc, void* data, hg_size_t size), which will
 * copy size bytes of the content of the memory pointed by data.
 */
static inline hg_return_t hg_proc_int_list_t(hg_proc_t proc, void* data)
{
    hg_return_t ret;
    int_list_t* list = (int_list_t* ) data;

    hg_size_t length = 0;
    int_list_t tmp   = NULL;
    int_list_t prev  = NULL;

    switch(hg_proc_get_op(proc)) {

        // The HG_ENCODE part is used when the proc handle is serializing
        // an existing object into a buffer. 
        case HG_ENCODE:
            tmp = *list;

            // find out the length of the list
            while(tmp != NULL) {
                tmp = tmp->next;
                length += 1;
            }

            // write the length
            ret = hg_proc_hg_size_t(proc, &length);
            if(ret != HG_SUCCESS)
                break;

            // write the list
            tmp = *list;
            while(tmp != NULL) {
                ret = hg_proc_int32_t(proc, &tmp->value);
                if(ret != HG_SUCCESS)
                    break;
                tmp = tmp->next;
            }

            break;

        // The HG_DECODE part is used when the proc handle is creating an
        // new object from the content of its buffer. 
        case HG_DECODE:
            // find out the length of the list
            ret = hg_proc_hg_size_t(proc, &length);
            if(ret != HG_SUCCESS)
                break;

            // loop and create list elements
            *list = NULL;
            while(length > 0) {
                tmp = (int_list_t)calloc(1, sizeof(*tmp));

                if(*list == NULL) {
                    *list = tmp;
                }

                if(prev != NULL) {
                    prev->next = tmp;
                }

                ret = hg_proc_int32_t(proc, &tmp->value);

                if(ret != HG_SUCCESS)
                    break;

                prev = tmp;
                length -= 1;
            }
            break;

        // The HG_FREE part is used when freeing the object, e.g.
        // when calling margo_free_input or margo_free_output.
        case HG_FREE:
            tmp = *list;

            while(tmp != NULL) {
                prev = tmp;
                tmp  = prev->next;
                free(prev);
            }

            ret = HG_SUCCESS;
    }
    return ret;
}

#endif