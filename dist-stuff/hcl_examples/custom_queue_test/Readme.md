## Try a distributed mxm-task queue with HCL
Assume the usecase, we have multiple ranks running on distributed memory machines, each rank has a queue of tasks and multiple execution threads inside. Task could be defined as a data tuple of matrices, for example:
```C
typedef struct mattup_stdarr_t {
    std::array<double, SIZE*SIZE> A;
    std::array<double, SIZE*SIZE> B;
    std::array<double, SIZE*SIZE> C;

    // constructor 1
    mattup_stdarr_t() {
        for (int i = 0; i < SIZE*SIZE; i++) {
            A[i] = 1.0;
            B[i] = 2.0;
            C[i] = 0.0;
        }
    }

    // serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        ar & A;
        ar & B; 
        ar & C;
    }

} mattup_stdarr_t;
```
A, B, C are the fixed-sized array with SIZE. Instead of creating separate queues on separate ranks, we try to create an HCL-global queue of mattup_stdarr_t type objects. In further, for the scanario as task-stealing, the current rank could easily get the task from other ranks, and put back later on.

<p align="left">
  <img src="./figures/usecase_mxm_task_queue.png" alt="An example with task-queues" width="700">
</p>