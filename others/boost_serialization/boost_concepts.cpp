#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////
////////// Global-Variables
///////////////////////////////////////////////////////////////////////////
std::stringstream ss;


///////////////////////////////////////////////////////////////////////////
////////// User-defined Classes
///////////////////////////////////////////////////////////////////////////
class animal {
    public:
        animal() = default;
        animal(int legs) : legs_(legs) {}
        int legs() const {return legs_; }
    
    private:
        friend class boost::serialization::access;

        // define serialize-function
        template <typename Archive>
        void serialize(Archive &ar, const unsigned int version) {
            std::cout << "\t    [DEBUG] calling sericalize()..." << std::endl;
            ar & legs_;
        }

        // a private attribute of the class
        int legs_;
};

///////////////////////////////////////////////////////////////////////////
////////// Util-Functions
///////////////////////////////////////////////////////////////////////////

void save() {
    std::cout << "\t save() is calling..." << std::endl;
    std::ofstream file{"io_archive_test.txt"};
    boost::archive::text_oarchive oa{file};
    int i = 10;
    oa << i;
}

void save_ss() {
    std::cout << "\t save_ss() is calling..." << std::endl;
    boost::archive::text_oarchive oa{ss};
    // create an instance of the class animal, having 4 legs
    animal s_a(4);
    oa << s_a;  // this operator will call serialize()
}

void load() {
    std::cout << "\t load() is calling..." << std::endl;
    std::ifstream file{"io_archive_test.txt"};
    boost::archive::text_iarchive ia{file};
    int i = 20;
    ia >> i;
    std::cout << "\t    load i: " << i << std::endl;
}

void load_ss() {
    std::cout << "\t load_ss() is calling..." << std::endl;
    boost::archive::text_iarchive ia{ss};
    // create an empty instance of the class animal
    animal l_a;
    ia >> l_a;
    std::cout << "\t    load a: " << l_a.legs() << std::endl;
}

///////////////////////////////////////////////////////////////////////////
////////// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv){

    /**
     * The main concept of Boost.Serialization is the archive. An archive is
     * a sequence of bytes that represent serialized C++ objects. Objects can be
     * added to an archive to serialize them and then later loaded from the archive.
     * In order to restore previously saved C++ objects, the same types are presumed.
     */
    std::cout << "1. Check the instance of boost::archive::text_oarchive." << std::endl;
    boost::archive::text_oarchive oa(std::cout);
    int i = 10;
    oa << i; std::cout << std::endl;
    std::cout << "------------------------------------" << std::endl;

    /**
     * The class boost::archive::text_oarchive serializes data as a text stream,
     * and the class boost::archive::text_iarchive restores data from such a text
     * stream. To use these classes, include the header files
     * boost/archive/text_iarchive.hpp and boost/archive/text_oarchive.hpp.
     */
    std::cout << "2. Calling save and load as a text stream by boost::archive." << std::endl;
    save();
    load();
    std::cout << "------------------------------------" << std::endl;

    std::cout << "3. Calling save_ss ad load_ss with the class - animal and boost::archive." << std::endl;
    save_ss();
    load_ss();
    std::cout << "------------------------------------" << std::endl;

    /**
     * In order to serialize objects of user-defined types, you must define the member
     * function serialize(). This function is called when the object is serialized to or
     * restored from a byte stream. Because serialize() is used for both serializing and restoring,
     * Boost.Serialization supports the operator operator & in addition to operator << and
     * operator >>. With operator& there is no need to distinguish between serializing and
     * restoring within serialize().
     * 
     * serialize() is automatically called any time an object is serialized or restored. It should
     * never be called explicitly and, thus, should be declared as private. If it is declared as
     * private, the class boost::serialization::access must be declared as a friend to
     * allow Boost.Serialization to access the member function.
     */

    return 0;
}