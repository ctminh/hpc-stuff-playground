#include <thallium/serialization/serialize.hpp>


// ================================================================================
// Struct Definition
// ================================================================================
struct mat_task {
    int size;
    double *A;  // ptr to the allocated matrix A
    double *B;  // ptr to the allocated matrix B
    double *C;  // ptr to the allocated matrix C - result
};

/**
 * Another class - Matrix
 * It's define as follows,
 */
struct Matrices {
    int size;
    double *A;
    double *B;
    double *C;

    // define the serialization inside
    template<class A>
    void serialize(A & ar, const unsigned int version){
        ar & size;

        for (int i = 0; i < size; i++){
            ar & this->A[i];
            ar & this->B[i];
            ar & this->C[i];
        }
    }
};