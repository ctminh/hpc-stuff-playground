#include "chameleon.h"
#include "chameleon_tools.h"
#include <unistd.h>
#include <sys/syscall.h>
#include <inttypes.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream> 
#include <sched.h>
#include <numeric>

#ifdef TRACE
#include "VT.h"
static int event_tool_task_create = -1;
static int event_tool_task_exec = -1;
#endif

#define TIMESTAMP(time_) 						\
  do {									\
      struct timespec ts;						\
      clock_gettime(CLOCK_MONOTONIC, &ts);				\
      time_ = ((double)ts.tv_sec) + (1.0e-9)*((double)ts.tv_nsec);		\
  } while(0)

void chameleon_t_print(char data[])
{
    struct timeval curTime;
    gettimeofday(&curTime, NULL);
    int milli = curTime.tv_usec / 1000;
    int micro_sec = curTime.tv_usec % 1000;
    char buffer [80];
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));
    char currentTime[84] = "";
    sprintf(currentTime, "%s.%03d.%03d", buffer, milli, micro_sec);
    printf("[CHAM_T] Timestamp-%s: %s\n", currentTime, data);
}

// global vars
cham_t_task_list_t tool_task_list;
std::vector<int64_t> arg_size_list(1000);
std::vector<double> cpu_freq_list(1000);

void *pack_tool_data(int32_t num_tasks, cham_t_task_list_t *tool_task_list, int32_t *buffer_size)
{
  // FORMAT:
    //      0. number of tasks    // 0
    //      Then for each task:
    //        TYPE_TASK_ID task_id; // 1
    //        int rank_belong;    // 2
    //        size_t size_data;   // 3
    //        double queue_time;  // 4
    //        double start_time;  // 5
    //        double end_time;    // 6
    //        double mig_time;    // 7
    //        double exe_time;    // 8
    //        bool migrated;      // 9
    //        int32_t arg_num;    // 10
    //        uintptr_t codeptr_ra;   // 11

    int total_size = sizeof(int32_t); // 0. number of tasks
    for (int i = 0; i < num_tasks; i++) {
      total_size += sizeof(TYPE_TASK_ID)  // 1. task_id
                    + sizeof(int)      // 2. rank_belong
                    + sizeof(size_t)   // 3. size_data
                    + sizeof(double)   // 4. queue_time
                    + sizeof(double)   // 5. start_time
                    + sizeof(double)   // 6. end_time
                    + sizeof(double)   // 7. mig_time
                    + sizeof(double)   // 8. exe_time
                    + sizeof(bool)     // 9. migrated
                    + sizeof(int32_t)  // 10. arg_num
                    + sizeof(uintptr_t); // 11. codeptr_ra
    }

    // allocate memory for transfer
    char *buff = (char *) malloc(total_size);
    char *cur_ptr = (char *)buff;

    // 0. number of tasks in this message
    ((int32_t *) cur_ptr)[0] = num_tasks;
    cur_ptr += sizeof(int32_t);

    // the first element in the list
    std::list<cham_t_task_info_t *>::iterator it = tool_task_list->task_list.begin();

    for (int i = 0; i < num_tasks; i++) {
      // 1. task_id
      std::advance(it, i);
      ((TYPE_TASK_ID *) cur_ptr)[0] = (*it)->task_id;
      cur_ptr += sizeof(TYPE_TASK_ID);

      // 2. rank_belong
      ((int *) cur_ptr)[0] = (*it)->rank_belong;
      cur_ptr += sizeof(int);

      // 3. size_data
      ((size_t *) cur_ptr)[0] = (*it)->arg_sizes[0];
      cur_ptr += sizeof(size_t);

      // 4. queue_time
      ((double *) cur_ptr)[0] = (*it)->queue_time;
      cur_ptr += sizeof(double);

      // 5. start_time
      ((double *) cur_ptr)[0] = (*it)->start_time;
      cur_ptr += sizeof(double);

      // 6. end_time
      ((double *) cur_ptr)[0] = (*it)->end_time;
      cur_ptr += sizeof(double);

      // 7. mig_time
      ((double *) cur_ptr)[0] = (*it)->mig_time;
      cur_ptr += sizeof(double);

      // 8. exe_time
      ((double *) cur_ptr)[0] = (*it)->exe_time;
      cur_ptr += sizeof(double);

      // 9. migrated
      ((bool *) cur_ptr)[0] = (*it)->migrated;
      cur_ptr += sizeof(bool);

      // 10. arg_num
      ((int32_t *) cur_ptr)[0] = (*it)->arg_num;
      cur_ptr += sizeof(int32_t);

      // 11. codeptr_ra
      ((uintptr_t *) cur_ptr)[0] = (*it)->codeptr_ra;
      cur_ptr += sizeof(uintptr_t);
    }

    *buffer_size = total_size;

    // check total_size
    printf("total size = %d, num_task = %d, cur_ptr = %p\n", total_size, num_tasks, cur_ptr);

    return buff;
}

void unpack_tool_data(void * buffer, int mpi_tag, int32_t *num_tasks, cham_t_task_list_t *tool_task_list){
  // current pointer position
  char *cur_ptr = (char*) buffer;

  // get number of tasks
  int n_tasks = ((int32_t *) cur_ptr)[0];
  cur_ptr += sizeof(int32_t);
  *num_tasks = n_tasks;
  tool_task_list->list_size = n_tasks;  // update the number of tasks
  
  printf("MPI_tag = %d, num_tasks = %d\n", mpi_tag, n_tasks);
}

#pragma region Local Helpers
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}
#pragma endregion Local Helpers

void chameleon_t_statistic(cham_t_task_list_t *tool_task_list, int mpi_rank){
  // write logfile
  std::ofstream outfile;
  outfile.open("./logfile.txt");

  int i = 1;
  
  printf("------------------------- Chameleon Statistics R%d ---------------------\n", mpi_rank);
  printf("TID Task_ID  arg_num  dat_size \t codeptr_ra \t\t\t\t\t  q_time \t\t\t\t\t  m_time \t\t\t\t\t m_des    s_time \t\t\t\t\t\t\t w_time \t\t e_time \t\t\t\t\t runtime\n"); //q_time  m_time  m_des  s_time  w_time  e_time  runtime\n");
  for (std::list<cham_t_task_info_t*>::iterator it=tool_task_list->task_list.begin(); it!=tool_task_list->task_list.end(); ++it) {
    double w_time = (*it)->start_time - (*it)->queue_time;
    if (w_time < 0)
      w_time = 0.0;
    printf("%2d  %5d  \t %2d \t\t\t\t\t\t\t\t " DPxMOD "  %f   %f  \t\t\t\t  %d \t\t  %f    %f \t %f  %f\n", 
            0,
            (*it)->task_id,
            (*it)->arg_num,
            DPxPTR((*it)->codeptr_ra),
            (*it)->queue_time,
            (*it)->mig_time,
            1,
            (*it)->start_time,
            w_time,
            (*it)->end_time,
            (*it)->exe_time);

    // write file
    std::string line = std::to_string((*it)->task_id) + ","
                      + std::to_string((*it)->arg_num) + ","
                      + std::to_string(arg_size_list[i]) + ","
                      + std::to_string((*it)->processed_freq) + ","
                      + std::to_string((*it)->exe_time) + "\n";
    outfile << line;
    i++;
  }

  // close file
  outfile.close();
}


double get_core_freq(int core_id){
  // read cpuinfo file
  std::string line;
  std::ifstream file ("/proc/cpuinfo");

  double freq = 0.0;
  int i = 0;

  if (file.is_open()){
    while (getline(file, line)){
      if (line.substr(0,7) == "cpu MHz"){
        if (i == core_id){
          std::string::size_type sz;
          freq = std::stod (line.substr(11,21), &sz);
          return freq;
        }
        else  i++;
      }
    }

    file.close();
  }
  else
  {
    printf("Unable to open file!\n");
  }
  
  return freq;
}
