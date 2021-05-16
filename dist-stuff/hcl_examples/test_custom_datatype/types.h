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
#include <vector>

#include <cereal/cereal.hpp> // for defering
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>


namespace bip=boost::interprocess;
const int SIZE = 100;

// ================================================================================
// Util-functions
// ================================================================================


// ================================================================================
// Struct Definition
// ================================================================================

/* Struct of a matrix-tuple type using std::array */
typedef struct Mattup_Stdarr_T {

    std::array<double, SIZE*SIZE> A;
    std::array<double, SIZE*SIZE> B;
    std::array<double, SIZE*SIZE> C;

    // constructor 1
    Mattup_Stdarr_T() {
        for (int i = 0; i < SIZE*SIZE; i++) {
            A[i] = 1.0;
            B[i] = 2.0;
            C[i] = 0.0;
        }
    }

    // serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        // serialize one by one
        // for (int i = 0; i < SIZE*SIZE; i++) {
        //     ar & A[i];
        //     ar & B[i];
        //     ar & C[i];
        // }

        // serialize a bulk of data
        ar & A; //.data();
        ar & B; //.data();
        ar & C; //.data();
    }
} Mattup_Stdarr_T;

/* Try a simple struct with a single double element */
typedef struct Single_DB_T {
    double a;

    // constructor 1
    Single_DB_T(): a(1.0) { }
    // constructor 2
    Single_DB_T(double val): a(val) { }

} Single_DB_T;

// try to get the serialization out of the struct define
#if defined(HCL_ENABLE_THALLIUM_TCP) || defined(HCL_ENABLE_THALLIUM_ROCE)
template<typename A>
void serialize(A &ar, Single_DB_T &a) {
    ar & a.a;
}
#endif

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