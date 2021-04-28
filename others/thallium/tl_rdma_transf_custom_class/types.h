#include <iostream>
#include <fstream>
#include <thallium/serialization/serialize.hpp>

class Point {
    private:
        double x;
        double y;
        double z;

    public:
        Point(double a=0.0, double b=0.0, double c=0.0)
            : x(a), y(b), z(c) {}

        static size_t size(){
            size_t ret = sizeof(double) * 3;
            return ret;
        }

        template<typename A>
        void serialize(A& ar) {
            ar & x;
            ar & y;
            ar & z;
        }
};

