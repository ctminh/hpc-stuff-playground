#ifndef MAT_TASK_TYPE_H
#define MAT_TASK_TYPE_H

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
const int SIZE = 128;

// ================================================================================
// Util-functions
// ================================================================================


// ================================================================================
// Struct Definition
// ================================================================================

/* Struct of a matrix-tuple type using std::array */
typedef struct MatTask_Type {
    int tid;
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;

    // constructor 1
    MatTask_Type(): tid(0), A(), B(), C() {}

    // constructor 2, init matrices with identical values
    MatTask_Type(int id, int val):
            tid(id),
            A(SIZE * SIZE, val),
            B(SIZE * SIZE, val),
            C(SIZE * SIZE, val) { }

    // constructor 3, init matrices with random values
    // TODO
    
    // overwrite operators
    bool operator==(const MatTask_Type &o) const {
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

    MatTask_Type &operator=(const MatTask_Type &other){
        tid = other.tid;
        A = other.A;
        B = other.B;
        C = other.C;
        return *this;
    }

    bool operator<(const MatTask_Type &o) const {

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
 
    bool operator>(const MatTask_Type &o) const {
        return !(*this < o);
    }
 
    bool Contains(const MatTask_Type &o) const {
        return *this == o;
    }

    // serialization for using rpc lib
#ifdef HCL_ENABLE_RPCLIB
    MSGPACK_DEFINE(A,B,C);
#endif

} MatTask_Type;

// serialization like thallium does
#if defined(HCL_ENABLE_THALLIUM_TCP) || defined(HCL_ENABLE_THALLIUM_ROCE)
    template<typename A>
    void serialize(A &ar, MatTask_Type &a) {
        ar & a.tid;
        ar & a.A;
        ar & a.B;
        ar & a.C;
    }
#endif

namespace std {
    template<>
    struct hash<MatTask_Type> {
        size_t operator()(const MatTask_Type &k) const {
            size_t hash_val = hash<int>()(k.A[0]);

            for (int i = 1; i < k.A.size(); ++i) {
                hash_val ^= hash<int>()(k.A[0]);
            }

            return hash_val;
        }
    };
}

#endif //MAT_TASK_TYPE_H