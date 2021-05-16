#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <iostream>
#include <string>


/**
 * A basic class - point with 3 double values inside.
 * The serialization/deserialization works with the
 * operators << or >>
 */
class point {

    private:
        friend class boost::serialization::access;
        double x;
        double y;
        double z;

        template<class A>
        void serialize(A& ar, const unsigned int version) {
             ar & x;
             ar & y;
             ar & z;
        }

    public:
        point(double a=0.0, double b=0.0, double c=0.0)
        : x(a), y(b), z(c) {}

        std::string to_str(){
            std::string ret = "(" + std::to_string(x) + ",";
            ret += std::to_string(y) + ",";
            ret += std::to_string(z) + ")";
            return ret;
        }
};

/**
 * Another class - Range
 * It's define as follows,
 */
template<class T>
struct Range {
    T begin, end;

    // define serialization for this struct
    template<class A>
    void serialize(A & ar, const unsigned int version){
        ar & begin;
        ar & end;
    }
};

template<class T, int Dim>
struct NRange {
    private:
        // point this to a friend of the class boost::serialization
        friend class boost::serialization::access;

        // a range of elements
        Range<T> ranges[Dim];

        // define the serialization for this struct
        template<class A>
        void serialize(A & ar, const unsigned int version){
            for (int i = 0; i < Dim; i++)
                ar & ranges[i];
        }


    public:

        // overload the operator [], to return the element
        // of the Range at the position i
        //  + note: & means we don't copy the value, this operator
        //          function would return the reference of the object.
        Range<T>& operator [](int i){
            return ranges[i];
        }

        // Todo: why need to define again the oper-function above with const
        const Range<T>& operator[](int i) const {
            return ranges[i];
        }
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