#include <fstream>

// include headers that implement a archive in simple text format
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

/////////////////////////////////////////////////////////////
// gps coordinate
//
// illustrates serialization for a simple type
//
class gps_postition {
    private:
        // define a friend class that means this class gps_postition
        // could be able to access somehow the priv-members of boost::serialization
        friend class boost::serialization::access;
        // When the class Archive corresponds to an output archive,
        //      + the & operator: is defined similar to <<.
        // Likewise, when the class Archive is a type of input archive
        //      + the & operator: is defined similar to >>.
        
        // the function serialize() would work with the template class Archive
        //      + this means we define this serialization function for the general
        //      + class. Not a specific one, and temporarily call it Archive class.
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version){
            ar & degrees;
            ar & minutes;
            ar & seconds;
        }

        // now is the private attributes
        int degrees;
        int minutes;
        float seconds;

    public:
        // constructors
        gps_postition(){};
        gps_postition(int d, int m, float s) :
            degrees(d), minutes(m), seconds(s)
            { };
};

int main(int argc, char *argv[]){
    // create and open a character archive for output
    // this construct an ofstream object that is not associated with any file.
    std::ofstream ofs("filename");

    // create a class instance - g
    const gps_postition g(35, 59, 24.567f);

    /** 
     * text_oarchive(std::ostream & os, unsigned int flags = 0);
     * 
     * Constructs an archive given an open stream as an argument and optional flags. For most
     * applications there will be no need to use flags. Flags are defined by enum archive_flags
     * enumerator. Multiple flags can be combined with the | operator. By default, archives prepend
     * output with initial data which helps identify them as archives produced by this system.
     * This permits a more graceful handling of the case where an attempt is made to load an archive
     * from an invalid file format. In addition to this, each type of archive might have its own information.
     * For example, native binary archives include information about sizes of native types and endianess to
     * gracefully handle the case where it has been erroneously assumed that such an archive is portable
     * across platforms. In some cases, where this extra overhead might be considered objectionable,
     * it can be suppressed with the no_header flag.
     * 
     * In some cases, an archive may alter (and later restore) the codecvt facet of the stream locale.
     * To suppress this action, include the no_codecvt flag.
     * 
     * XML archives contain nested tags signifying the start and end of data fields. These tags are normally
     * checked for agreement with the object name when data is loaded. If a mismatch occurs an exception is
     * thrown. It's possible that this may not be desired behavior. To suppress this checking of XML tags,
     * use no_xml_tag_checking flag.
     * 
     * ~text_oarchive();
     * 
     * Destructor for an archive. This should be called before the stream is closed. It restores any altered
     * stream facets to their state before the archive was opened.
     */

    // save data to archive 
    boost::archive::text_oarchive oa(ofs);
    // write the gps class instance to archive
    oa << g;
    // note: archive and stream closed when destructors are called

    // ... some time later restore the class instance to its original state
    gps_postition new_g;
    // create and open an archive for input
    std::ifstream ifs("filename");
    boost::archive::text_iarchive ia(ifs);
    // read the class state from archive
    ia >> new_g;
    // note: archive and stream closed when destructors are called

    return 0;
}


/* Notations 
 * 
 * In the following descriptions
 *  + SA is an type modeling the Saving Archive Concept.
 *  + sa is an instance of type SA.
 *  + LA is an type modeling the Loading Archive Concept.
 *  + la is an instance of type LA.
 *  + T is an Serializable Type.
 *  + x is an instance of type T Type.
 *  + u,v is a pointer to an instance of type T.
 *  + count is an instance of a type that can be converted to std::size_t.
 * 
 * Saving Archive Concept
 * Associated Types
 * 
 * Intuitively, a type modeling this concept will generate a sequence of bytes corresponding to an
 * arbitrary set of C++ data structures. Each type modeling the Saving Archive concept (SA) may be
 * associated with another type modeling the Loading Archive Concept(LA). This associated type will
 * perform the inverse operation. That is, given a sequence of bytes generated by SA, it will generate
 * a set of C++ data structures that is equivalent to the original. The notion of equivalence is defined
 * by the implementations of the pair of archives and the way the data are rendered serializable.
 */