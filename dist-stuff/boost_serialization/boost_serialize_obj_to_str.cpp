#include <mpi.h>
#include <fstream>
#include <random>
#include <cmath>
#include "types.h"

///////////////////////////////////////////////////////////////////////////
////////// Global-Variables
///////////////////////////////////////////////////////////////////////////
std::stringstream ss_point;
std::stringstream ss_nrange;
std::stringstream ss_matrices;

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

///////////////////////////////////////////////////////////////////////////
////////// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){

    // create a point-object
    point P1(1.0, 1.0, 1.0);
    point P2(2.0, 2.0, 2.0);
    point P3(3.0, 3.0, 3.0);

    // check the point
    std::cout << "1. Init points and check the values:" << std::endl;
    std::cout << "\tP1: " << P1.to_str() << std::endl;
    std::cout << "\tP2: " << P2.to_str() << std::endl;
    std::cout << "\tP3: " << P3.to_str() << std::endl;
    std::cout << "------------------------------------\n" << std::endl;

    // serialize the points
    std::cout << "2. Serialize the point-objects:" << std::endl;
    boost::archive::text_oarchive oa{ss_point};
    oa << P1 << P2 << P3;
    std::string conv_ss_point_to_str = ss_point.str();
    std::cout << "\tss_point to str: ";
    std::cout << conv_ss_point_to_str << std::endl;
    std::cout << "\tsize = " << conv_ss_point_to_str.size() << " bytes" << std::endl;
    std::cout << "------------------------------------\n" << std::endl;

    // deserialize the points;
    std::cout << "3. Deserialize the point-objects:" << std::endl;
    point des_P1;
    point des_P2;
    point des_P3;
    boost::archive::text_iarchive ia{ss_point};
    // ia >> des_P1;
    // std::cout << "\tdes_P1: " << des_P1.to_str() << std::endl;
    // ia >> des_P2;
    // std::cout << "\tdes_P2: " << des_P2.to_str() << std::endl;
    // ia >> des_P3;
    // std::cout << "\tdes_P3: " << des_P3.to_str() << std::endl;
    ia >> des_P1 >> des_P2 >> des_P3;
    std::cout << "\tdes_P1: " << des_P1.to_str() << std::endl;
    std::cout << "\tdes_P2: " << des_P2.to_str() << std::endl;
    std::cout << "\tdes_P3: " << des_P3.to_str() << std::endl;
    std::cout << "------------------------------------\n" << std::endl;

    ////////////////////////////////////////////////////////////////
    // declare a range of numbers with the struct NRange
    const int N = 5;
    NRange<double, N> range;

    // init values for this range
    for (int i = 0; i < N; i++){
        range[i].begin = double(i);
        range[i].end = double(i);
    }

    // serialize the range
    std::cout << "4. Serialize the nrange-objects:" << std::endl;
    boost::archive::text_oarchive oa_nrange{ss_nrange};
    oa_nrange << range;
    std::string conv_ss_nrange_to_str = ss_nrange.str();
    std::cout << "\tss_nrange to str: ";
    std::cout << conv_ss_nrange_to_str << std::endl;
    std::cout << "\tsize = " << conv_ss_nrange_to_str.size() << " bytes" << std::endl;
    std::cout << "------------------------------------\n" << std::endl;

    // deserialize the range
    std::cout << "5. Deserialize the nrange-objects:" << std::endl;
    NRange<double, N> des_range;
    boost::archive::text_iarchive ia_nrange{ss_nrange};
    ia_nrange >> des_range;
    for (int i = 0; i < N; i++){
        std::cout << "\tR" << i << ": begin=" << des_range[i].begin << ", end=" << des_range[i].end << std::endl;
    }
    std::cout << "------------------------------------\n" << std::endl;

    // serialize the matrices
    int mat_size = 10;
    Matrices Ms;
    Ms.size = mat_size*mat_size;
    Ms.A = new double[mat_size * mat_size]; // Ms.A is a pointer which is pointing to this new allocated mem
    Ms.B = new double[mat_size * mat_size]; // and similar to Ms.B, Ms.C, ...
    Ms.C = new double[mat_size * mat_size];
    // initialize values for the matrices
    initialize_matrix_rando(Ms.A, mat_size);
    initialize_matrix_rando(Ms.B, mat_size);
    initialize_matrix_zeros(Ms.C, mat_size);
    // serialize these objects
    std::cout << "6. Serialize the matrices-objects:" << std::endl;
    boost::archive::text_oarchive oa_matrices{ss_matrices};
    oa_matrices << Ms;
    std::string conv_ss_matrices_to_str = ss_matrices.str();
    // std::cout << "\tss_matrices to str: ";
    // std::cout << conv_ss_matrices_to_str << std::endl;
    std::cout << "\tsize = " << conv_ss_matrices_to_str.size() << " bytes" << std::endl;
    std::cout << "------------------------------------\n" << std::endl;

    // deserialize the matrices
    std::cout << "7. Deserialize the matrices-objects:" << std::endl;
    Matrices des_Ms;
    des_Ms.size = mat_size*mat_size;
    des_Ms.A = new double[mat_size * mat_size];
    des_Ms.B = new double[mat_size * mat_size];
    des_Ms.C = new double[mat_size * mat_size];
    boost::archive::text_iarchive ia_matrices{ss_matrices};
    ia_matrices >> des_Ms;
    for (int i = 0; i < mat_size; i++){
        for (int j = 0; j < mat_size; j++){
            std::cout << des_Ms.A[i*mat_size + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------\n" << std::endl;

    // free the memory
    delete[] Ms.A;
    delete[] Ms.B;
    delete[] Ms.C;
    delete[] des_Ms.A;
    delete[] des_Ms.B;
    delete[] des_Ms.C;

    return 0;
}