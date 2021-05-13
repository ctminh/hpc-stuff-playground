#ifndef HCL_UTIL_H
#define HCL_UTIL_H

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <queue>
#include <fstream>
#include <atomic>
#include <random>
#include <array>

namespace bip=boost::interprocess;
const int SIZE = 10;

// ================================================================================
// Util-functions
// ================================================================================

void initialize_matrix_rando(double *mat_ptr, int size){
    double low_bnd = 0.0;
    double upp_bnd = 10.0;
    std::uniform_real_distribution<double> ur_dist(low_bnd, upp_bnd);
    std::default_random_engine dre;
    for (int i = 0; i < size*size; i++){
        mat_ptr[i] = ur_dist(dre);
    }
}

void initialize_matrix_zeros(double *mat_ptr, int size){
    for (int i = 0; i < size*size; i++){
        mat_ptr[i] = 0.0;
    }
}

void compute_mxm(double *a, double *b, double *c, int size){
    
    // main loop to compute as a serial way
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            c[i*size + j] = 0;
            for (int k = 0; k < size; k++) {
                c[i*size + j] += a[i*size + k] * b[k*size + j];
            }
        }
    }
}

void mxm_kernel(double *A, double *B, double *C, int size, int i){

    // call the compute entry
    compute_mxm(A, B, C, size);
}

// ================================================================================
// Struct Definition
// ================================================================================

/* Struct of double pointers */
typedef struct double_ptr_t {
    int size;
    double *A;  // ptr to the allocated matrix A
    double *B;  // ptr to the allocated matrix B
    double *C;  // ptr to the allocated matrix C - result

    // Constructor 1
    double_ptr_t(){
        A = new double[10*10];
        B = new double[10*10];
        C = new double[10*10];
    }

    // Constructor 2
    double_ptr_t(int s){
        size = s;
        A = new double[s*s];
        B = new double[s*s];
        C = new double[s*s];
        initialize_matrix_rando(A, s);
        initialize_matrix_rando(B, s);
        initialize_matrix_zeros(C, s);
    }

    // Serialization
    template <typename Archive>
    void serialize(Archive &ar) {
        // for (int i = 0; i < 10*10; i++){
        //     ar & A[i];
        //     ar & B[i];
        //     ar & C[i];
        // }
        boost::serialization::make_array<double>(A, 100);
        boost::serialization::make_array<double>(B, 100);
        boost::serialization::make_array<double>(C, 100);
    }

}double_ptr_t;


/* Struct of double array */
typedef struct double_arr_t {
    double A[SIZE*SIZE];
    double B[SIZE*SIZE];
    double C[SIZE*SIZE];

    // Constructor 1
    double_arr_t(){
        double low_bnd = 0.0;
        double upp_bnd = 10.0;
        std::uniform_real_distribution<double> ur_dist(low_bnd, upp_bnd);
        std::default_random_engine dre;
        for (int i = 0; i < SIZE*SIZE; i++){
            A[i] = ur_dist(dre);
            B[i] = ur_dist(dre);
            C[i] = 0.0;
        }
    }

    // serialization
    template <typename Archive>
    void serialize(Archive &ar) const {
        for (int i = 0; i < SIZE*SIZE; i++) {
            ar & A[i];
            ar & B[i];
            ar & C[i];
        }
    }

}double_arr_t;


/* Struct of mixed types */
typedef struct general_task_t {
    int id;
    int32_t idx_image = 0;
    int32_t arg_num;
    std::vector<void *> arg_hst_pointers;
    std::vector<int64_t> arg_sizes;
}general_task_t;


/* Struct of a single db-array type */
typedef struct single_db_arr_t {
    double x[100];

    // constructor 1
    single_db_arr_t(){
        for (int i = 0; i < 100; i++)
            x[i] = (double) i;
    }

    // constructor 2
    single_db_arr_t(double a){
        for (int i = 0; i < 100; i++)
            x[i] = a;
    }

    // serialization
    template <typename Archive>
    void serialize(Archive &ar) const {
        for (int i = 0; i < 100; i++) {
            ar & x[i];
        }
        // ar & boost::serialization::make_array<double>(x);
    }

}single_db_arr_t;


/* Struct of a single db-array type using std::array */
typedef struct single_db_stdarr_t {

    std::array<double,62500> a;

    // constructor 1
    single_db_stdarr_t() {
        for (int i = 0; i < 62500; i++) {
            a[i] = (double) i;
        }
    }

    // serialization
    template<typename A>
    void serialize(A& ar) const {
        for (int i = 0; i < 62500; i++) {
            ar & a[i];
        }
        // boost::serialization::make_nvp("elems", (static_cast<void *>(a.data()));
        // ar & a.data();
    }
};

// ================================================================================
// HCL-author-defined Types for Testing
// ================================================================================

struct KeyType{
    size_t a;
    KeyType():a(0){}
    KeyType(size_t a_):a(a_){}

#if HCL_ENABLE_RPCLIB
    MSGPACK_DEFINE(a);
#endif

    /* equal operator for comparing two Matrix. */
    bool operator==(const KeyType &o) const {
        return a == o.a;
    }
    KeyType& operator=( const KeyType& other ) {
        a = other.a;
        return *this;
    }
    bool operator<(const KeyType &o) const {
        return a < o.a;
    }
    bool Contains(const KeyType &o) const {
        return a==o.a;
    }
    
    template<typename A>
    void serialize(A& ar) const {
        ar & a;
    }
};

struct MappedType{

    std::string a;

    MappedType() {
        a = "";
        a.resize(1000000, 'c');
    }

    MappedType(std::string a_) {
        a = a_;
    }

#if HCL_ENABLE_RPCLIB
    MSGPACK_DEFINE(a);
#endif

    /* equal operator for comparing two Matrix. */
    bool operator==(const MappedType &o) const {
        if (a == o.a) {
            return false;
        }

        return true;
    }
    MappedType& operator=( const MappedType& other ) {
        a = other.a;
        return *this;
    }
    
    template<typename A>
    void serialize(A& ar) const {
        ar & a;
    }
};

const int MAX = 26;
std::string printRandomString(int n) {
    char alphabet[MAX] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                           'h', 'i', 'j', 'k', 'l', 'm', 'n',
                           'o', 'p', 'q', 'r', 's', 't', 'u',
                           'v', 'w', 'x', 'y', 'z' };

    std::string res = "";
    for (int i = 0; i < n; i++)
        res = res + alphabet[rand() % MAX];

    return res;
}

namespace std {
    template<>
    struct hash<KeyType> {
        int operator()(const KeyType &k) const {
            return k.a;
        }
    };
}

void bt_sighandler(int sig, struct sigcontext ctx) {

    void *trace[16];
    char **messages = (char **)NULL;
    int i, trace_size = 0;

    if (sig == SIGSEGV)
        printf("Got signal %d, faulty address is %p, from %p\n", sig, ctx.cr2, ctx.rip);
    else
        printf("Got signal %d\n", sig);

    trace_size = backtrace(trace, 16);

    /* overwrite sigaction with caller's address */
    trace[1] = (void *)ctx.rip;
    messages = backtrace_symbols(trace, trace_size);

    /* skip first stack frame (points here) */
    printf("[bt] Execution path:\n");
    for (i=1; i<trace_size; ++i) {
        printf("[bt] #%d %s\n", i, messages[i]);

        /* find first occurence of '(' or ' ' in message[i] and assume
         * everything before that is the file name. (Don't go beyond 0 though
         * (string terminator)*/
        size_t p = 0;
        while(messages[i][p] != '(' && messages[i][p] != ' '
              && messages[i][p] != 0)
            ++p;

        char syscom[256];
        sprintf(syscom,"addr2line %p -e %.*s", trace[i], p, messages[i]);
        //last parameter is the file name of the symbol
        system(syscom);
    }

    exit(0);
}

void SetSignal(){
    struct sigaction sa;

    sa.sa_handler = reinterpret_cast<__sighandler_t>(bt_sighandler);
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGUSR1, &sa, NULL);
    sigaction(SIGABRT, &sa, NULL);
}
#endif //HCL_UTIL_H