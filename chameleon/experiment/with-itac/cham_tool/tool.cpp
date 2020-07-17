#include "tool.h"

#define TASK_TOOL_SAMPLE_DATA_SIZE 10
#define THRESHOLD_OFFLOAD_TASKS 100

static cham_t_set_callback_t cham_t_set_callback;
static cham_t_get_callback_t cham_t_get_callback;
static cham_t_get_rank_data_t cham_t_get_rank_data;
static cham_t_get_thread_data_t cham_t_get_thread_data;
static cham_t_get_rank_info_t cham_t_get_rank_info;
static cham_t_get_task_data_t cham_t_get_task_data;
// static cham_t_get_unique_id_t cham_t_get_unique_id;

//================================================================
// Variables
//================================================================

//================================================================
// Additional functions
//================================================================

int compare( const void *pa, const void *pb ){
    const int *a = (int *) pa;
    const int *b = (int *) pb;
    if(a[0] == b[0])
        return a[0] - b[0];
    else
        return a[1] - b[1];
}

//================================================================
// Callback Functions
//================================================================ 
static void
on_cham_t_callback_thread_init(
    cham_t_data_t *thread_data)
{
    int rank_info       = cham_t_get_rank_info()->comm_rank;
    thread_data->value = syscall(SYS_gettid);
}

static void
on_cham_t_callback_thread_finalize(
    cham_t_data_t *thread_data)
{
    int rank_info       = cham_t_get_rank_info()->comm_rank;
    thread_data->value = syscall(SYS_gettid);
}

static void
on_cham_t_callback_task_create(
    cham_migratable_task_t * task,
    std::vector<int64_t> arg_sizes,
    intptr_t codeptr_ra)
{
    int rank_info       = cham_t_get_rank_info()->comm_rank;
    TYPE_TASK_ID internal_task_id        = chameleon_get_task_id(task);
    double q_time = omp_get_wtime();
    
    // create custom data structure and use task_data as pointer
    cham_t_task_info_t * cur_task  = (cham_t_task_info_t*) malloc(sizeof(cham_t_task_info_t));
    cur_task->task_id           = internal_task_id;
    cur_task->rank_belong       = rank_info;
    cur_task->queue_time        = q_time;
    cur_task->codeptr_ra        = codeptr_ra;
    cur_task->arg_num           = chameleon_get_arg_num(task);

    arg_size_list[internal_task_id] = arg_sizes[0];

    // printf("R%d: Task%d-", rank_info, internal_task_id);
    // for (int i = 0; i < cur_task->arg_num; i++){
    //     printf("arg%d(size-%ldKB) ", i, arg_sizes[i]);
    // }
    // printf("\n");

    // add task to the queue
    tool_task_list.push_back(cur_task);
}

static void
on_cham_t_callback_task_schedule(
    cham_migratable_task_t * task,                   // opaque data type for internal task
    cham_t_task_flag_t task_flag,
    cham_t_data_t *task_data,
    cham_t_task_schedule_type_t schedule_type,
    cham_migratable_task_t * prior_task,             // opaque data type for internal task
    cham_t_task_flag_t prior_task_flag,
    cham_t_data_t *prior_task_data)
{
    TYPE_TASK_ID task_id = chameleon_get_task_id(task);
    int rank = cham_t_get_rank_info()->comm_rank;
}

static int32_t
on_cham_t_callback_determine_local_load(
    TYPE_TASK_ID* task_ids_local,
    int32_t num_tasks_local,
    TYPE_TASK_ID* task_ids_local_rep,
    int32_t num_tasks_local_rep,
    TYPE_TASK_ID* task_ids_stolen,
    int32_t num_tasks_stolen,
    TYPE_TASK_ID* task_ids_stolen_rep,
    int32_t num_tasks_stolen_rep)
{
    int rank_info = cham_t_get_rank_info()->comm_rank;
    return num_tasks_local;
}

static cham_t_migration_tupel_t*
on_cham_t_callback_select_tasks_for_migration(
    const int32_t* load_info_per_rank,
    TYPE_TASK_ID* task_ids_local,
    int32_t num_tasks_local,
    int32_t num_tasks_stolen,
    int32_t* num_tuples)
{
    cham_t_rank_info_t *rank_info  = cham_t_get_rank_info();
    cham_t_migration_tupel_t* task_migration_tuples = NULL;
    *num_tuples = 0;

    if (num_tasks_local > 0){
        task_migration_tuples = (cham_t_migration_tupel_t *)malloc(sizeof(cham_t_migration_tupel_t));   // allocate mem for the tuple
        int tmp_sorted_array[rank_info->comm_size][2];  // sort loads by rank
        int i;
        for (i = 0; i < rank_info->comm_size; i++){
            tmp_sorted_array[i][0] = i; // contain rank index
            tmp_sorted_array[i][1] = load_info_per_rank[i]; // contain load per rank
        }
        // check the values
        // for(i = 0; i < rank_info->comm_size; ++i)
        //     printf("Rank-%d, load=%d\n", tmp_sorted_array[i][0], tmp_sorted_array[i][1]);

        qsort(tmp_sorted_array, rank_info->comm_size, sizeof tmp_sorted_array[0], compare);

        // check after sorting
        // for(i = 0; i < rank_info->comm_size; ++i)
        //     printf("Rank-%d, load=%d\n", tmp_sorted_array[i][0], tmp_sorted_array[i][1]);

        int min_val = load_info_per_rank[tmp_sorted_array[0][0]];   // load rank 0
        int max_val = load_info_per_rank[tmp_sorted_array[rank_info->comm_size-1][0]];  // load rank 1
        int load_this_rank = load_info_per_rank[rank_info->comm_rank];

        if (max_val > min_val){
            int pos = 0;
            for (i = 0; i < rank_info->comm_size; i++){
                if (tmp_sorted_array[i][0] == rank_info->comm_rank){
                    pos = i;
                    break;
                }
            }

            // only offload if on the upper side
            if((pos+1) >= ((double)rank_info->comm_size/2.0))
            {
                int other_pos = rank_info->comm_size-pos;
                // need to adapt in case of even number
                if(rank_info->comm_size % 2 == 0)
                    other_pos--;
                int other_idx = tmp_sorted_array[other_pos][0];
                int other_val = load_info_per_rank[other_idx];
                // calculate ration between those two and just move if over a certain threshold
                double ratio = (double)(load_this_rank-other_val) / (double)load_this_rank;
                if(other_val < load_this_rank && ratio > 0.5) {
                    double mig_time = omp_get_wtime();
                    task_migration_tuples[0].task_id = task_ids_local[0];
                    task_migration_tuples[0].rank_id = other_idx;
                    tool_task_list.set_migrated_time(task_ids_local[0], mig_time);
                    *num_tuples = 1;
                }
            }
        }

        if(*num_tuples <= 0){
            free(task_migration_tuples);
            task_migration_tuples = NULL;
        }
    }

    return task_migration_tuples;
}

static void
on_cham_t_callback_select_num_tasks_to_offload(
    std::vector<int32_t>& num_tasks_to_offload_per_rank,
    std::vector<int32_t>& load_info_per_rank,
    int32_t num_tasks_local,
    int32_t num_tasks_stolen,
    int32_t num_tasks_offloaded)
{
    // get rank info
    cham_t_rank_info_t *rank_info = cham_t_get_rank_info();
    // sort load and idx by load
    std::vector<size_t> tmp_sorted_idx = sort_indexes(load_info_per_rank);
    // choose the rank with min_load, max_load and load of the current rank
    double min_val      = (double) load_info_per_rank[tmp_sorted_idx[0]];
    double max_val      = (double) load_info_per_rank[tmp_sorted_idx[rank_info->comm_size-1]];
    double cur_load     = (double) load_info_per_rank[rank_info->comm_rank];
    // init & calculate ratio
    double ratio_lb     = 0.0; // 1 = high imbalance, 0 = no imbalance
    if (max_val > 0) {
        ratio_lb = (double)(max_val-min_val) / (double)max_val;
    }

    // check absolute condition
    static double min_abs_imbalance_before_migration = 2;
    static double min_rel_load_imbalance_before_migration = 0.05;
    static double percentage_diff_tasks_to_migrate = 1;

    if((cur_load-min_val) < min_abs_imbalance_before_migration) // < 2
        return;

    if(ratio_lb >= min_rel_load_imbalance_before_migration) {   // < 0.05
        // determine index of the current rank int the array of sorted load index
        int cur_rank_index = std::find(tmp_sorted_idx.begin(), tmp_sorted_idx.end(), rank_info->comm_rank) - tmp_sorted_idx.begin();

        /* Overview about the algorithm for migration
            all ranks would be sorted by their load: e.g., 6 ranks are sorted by their load [1][4][0][2][5][3]
            R5 with the idx = 4, R2 with the idx = 3
            so, the migrated-partner of R5 = 6 - 4(idx) - 1 = 1 (R5 will migrate tasks to R with idx=1, R4)
                the migrated-partner of R2 = 6 - 3(idx) - 1 = 2 (R2 will migrate tasks to R with idx=2, R0)
         */

        if(cur_rank_index >= ((double)rank_info->comm_size/2.0))    // if the cur_rank_index >= total_rank/2 => this rank have the big load, and need to migrate some
        {
            int migrated_rank_idx       = rank_info->comm_size - cur_rank_index - 1;    // the migrated-partner of this cur_rank
            int migrated_rank           = tmp_sorted_idx[migrated_rank_idx];            // get rank number
            double migrated_rank_load    = (double) load_info_per_rank[migrated_rank];  // get load of the migrated rank (receiver)

            double load_diff = cur_load - migrated_rank_load;

            // check absolute condition
            if(load_diff < min_abs_imbalance_before_migration)  // < 2
                return;

            double ratio = load_diff / (double)cur_load;
            // printf("R%d - check the total offloaded tasks = %d\n", rank_info->comm_rank, num_tasks_offloaded);
            if(migrated_rank_load < cur_load && ratio >= min_rel_load_imbalance_before_migration && num_tasks_offloaded <= THRESHOLD_OFFLOAD_TASKS) { // >= 0.05 (5%)
                int num_tasks = (int)(load_diff * percentage_diff_tasks_to_migrate);    // = the load_diff * 100%, e.g., load_diff=100, there must be 10 tasks from cur_rank ->(move)-> migrated_rank
                if(num_tasks < 1)
                    num_tasks = 1;

                num_tasks_to_offload_per_rank[migrated_rank] = num_tasks;
            }
        }
    }

}

static int32_t
on_cham_t_callback_change_freq_for_execution(
    cham_migratable_task_t * task,
    int32_t load_info_per_rank,
    int32_t total_created_tasks_per_rank
)
{
    int half_processed_per_rank = total_created_tasks_per_rank / 2; // 50% processed load
    int32_t noise_time = 0;
    int rank = cham_t_get_rank_info()->comm_rank;
    if (load_info_per_rank <= half_processed_per_rank && load_info_per_rank != 0){
        noise_time = 235050;    // make noise 50% slower for a mxm-task with size = 1024
        // noise_time = 507850;    // make noise 50% slower for non-uni tasks with size_range = {128, 256, 512, 1024, 2048}
    }

    return noise_time;
}

static void
on_cham_t_callback_task_processed(
    cham_migratable_task_t * task)
{
    TYPE_TASK_ID task_id = chameleon_get_task_id(task);
    double start_time = omp_get_wtime();
    tool_task_list.set_start_time(task_id, start_time);

    // get core info
    int core_id = sched_getcpu();
    double core_freq = get_core_freq(core_id);
    printf("Task-%d: Core ID = %d, freq = %f\n", task_id, core_id, core_freq);
    tool_task_list.set_processed_freq(task_id, core_freq);
}

static void
on_cham_t_callback_task_end(
    cham_migratable_task_t * task)
{
    TYPE_TASK_ID task_id = chameleon_get_task_id(task);
    int rank = cham_t_get_rank_info()->comm_rank;
    double end_time = omp_get_wtime();
    tool_task_list.set_end_time(task_id, end_time); 
}

//================================================================
// Start Tool & Register Callbacks
//================================================================
#define register_callback_t(name, type)                                         \
do{                                                                             \
    type f_##name = &on_##name;                                                 \
    if (cham_t_set_callback(name, (cham_t_callback_t)f_##name) == cham_t_set_never)   \
        printf("0: Could not register callback '" #name "'\n");                 \
} while(0)

#define register_callback(name) register_callback_t(name, name##_t)

int cham_t_initialize(
    cham_t_function_lookup_t lookup,
    cham_t_data_t *tool_data)
{
    printf("Calling register_callback...\n");
    cham_t_set_callback = (cham_t_set_callback_t) lookup("cham_t_set_callback");
    // cham_t_get_callback = (cham_t_get_callback_t) lookup("cham_t_get_callback");
    cham_t_get_rank_data = (cham_t_get_rank_data_t) lookup("cham_t_get_rank_data");
    cham_t_get_thread_data = (cham_t_get_thread_data_t) lookup("cham_t_get_thread_data");
    cham_t_get_rank_info = (cham_t_get_rank_info_t) lookup("cham_t_get_rank_info");
    // cham_t_get_task_data = (cham_t_get_task_data_t) lookup("cham_t_get_task_data");

    // register_callback(cham_t_callback_thread_init);
    // register_callback(cham_t_callback_thread_finalize);
    register_callback(cham_t_callback_task_create);
    // register_callback(cham_t_callback_task_schedule);
    register_callback(cham_t_callback_task_end);
    register_callback(cham_t_callback_task_processed);
    // register_callback(cham_t_callback_encode_task_tool_data);
    // register_callback(cham_t_callback_decode_task_tool_data);
    // register_callback(cham_t_callback_sync_region);
    // register_callback(cham_t_callback_determine_local_load);
    // register_callback(cham_t_callback_select_tasks_for_migration);
    // register_callback(cham_t_callback_change_freq_for_execution);

    // Priority is cham_t_callback_select_tasks_for_migration (fine-grained)
    // if not registered cham_t_callback_select_num_tasks_to_offload is used (coarse-grained)
    register_callback(cham_t_callback_select_num_tasks_to_offload);
    // register_callback(cham_t_callback_select_num_tasks_to_replicate);

    // cham_t_rank_info_t *r_info  = cham_t_get_rank_info();
    // cham_t_data_t * r_data      = cham_t_get_rank_data();
    // r_data->value               = r_info->comm_rank;

    return 1; //success
}

void cham_t_finalize(cham_t_data_t *tool_data)
{
    int rank = cham_t_get_rank_info()->comm_rank;
    if (rank == 0){
        chameleon_t_statistic(&tool_task_list, rank);
        // send the ordering signal to other ranks
        int next_print = 1;
        MPI_Send(&next_print, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }else{
        // wait for receiving the signal first
        int received_signal;
        MPI_Recv(&received_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("\n\n");
        
        if (received_signal == 1)
            chameleon_t_statistic(&tool_task_list, rank);
    }
}

#ifdef __cplusplus
extern "C" {
#endif
cham_t_start_tool_result_t* cham_t_start_tool(unsigned int cham_version)
{
    printf("Starting tool with Chameleon Version: %d\n", cham_version);

    static cham_t_start_tool_result_t cham_t_start_tool_result = {&cham_t_initialize, &cham_t_finalize, 0};

    return &cham_t_start_tool_result;
}
#ifdef __cplusplus
}
#endif