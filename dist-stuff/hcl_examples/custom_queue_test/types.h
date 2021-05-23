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

#include <cereal/cereal.hpp> // for defer
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>


namespace bip=boost::interprocess;
const int SIZE = 512;

// ================================================================================
// Util-functions
// ================================================================================


// ================================================================================
// Struct Definition
// ================================================================================

/* Struct of a matrix-tuple type using std::array */
typedef struct Mattup_StdArr_Type {
    int tid;
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;

    // constructor 1
    Mattup_StdArr_Type(): A(), B(), C() {}

    // constructor 2
    Mattup_StdArr_Type(int id, int val):
            tid(id),
            A(SIZE * SIZE, val),
            B(SIZE * SIZE, val),
            C(SIZE * SIZE, val) { }
    
    // overwrite operators
    bool operator==(const Mattup_StdArr_Type &o) const {
        if (o.tid != tid) return false;
        if (o.A.size() != A.size()) return false;
        if (o.B.size() != B.size()) return false;
        if (o.C.size() != C.size()) return false;
        
        for (int i = 0; i < SIZE*SIZE; ++i) {
            if (o.A[i] != A[i]) return false;
            if (o.B[i] != B[i]) return false;
            if (o.C[i] != C[i]) return false;
        }

        return true;
    }

    Mattup_StdArr_Type &operator=(const Mattup_StdArr_Type &other){
        tid = other.tid;
        A = other.A;
        B = other.B;
        C = other.C;
        return *this;
    }

    bool operator<(const Mattup_StdArr_Type &o) const {

        if (o.A.size() < A.size()) return false;
        if (o.A.size() > A.size()) return true;
        if (o.B.size() < B.size()) return false;
        if (o.B.size() > B.size()) return true;
        if (o.C.size() < C.size()) return false;
        if (o.C.size() > C.size()) return true;
        
        for (int i = 0; i < SIZE*SIZE; ++i) {
            if (o.A[i] < A[i]) return false;
            if (o.A[i] > A[i]) return true;
            if (o.B[i] < B[i]) return false;
            if (o.B[i] > B[i]) return true;
            if (o.C[i] < C[i]) return false;
            if (o.C[i] > C[i]) return true;
        }

        return false;
    }
 
    bool operator>(const Mattup_StdArr_Type &o) const {
        return !(*this < o);
    }
 
    bool Contains(const Mattup_StdArr_Type &o) const {
        return *this == o;
    }

    // serialization for using rpc lib
#ifdef HCL_ENABLE_RPCLIB
    MSGPACK_DEFINE(A,B,C);
#endif

} Mattup_StdArr_Type;

// serialization for using thallium
#if defined(HCL_ENABLE_THALLIUM_TCP) || defined(HCL_ENABLE_THALLIUM_ROCE)
    template<typename A>
    void serialize(A &ar, Mattup_StdArr_Type &a) {
        ar & a.tid;
        ar & a.A;
        ar & a.B;
        ar & a.C;
    }
#endif

namespace std {
    template<>
    struct hash<Mattup_StdArr_Type> {
        size_t operator()(const Mattup_StdArr_Type &k) const {
            size_t hash_val = hash<int>()(k.A[0]);

            for (int i = 1; i < k.A.size(); ++i) {
                hash_val ^= hash<int>()(k.A[0]);
            }
            
            return hash_val;
        }
    };
}


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