// widget.hpp
#ifndef __WIDGET_H
#define __WIDGET_H

#include <string>

// ignore this line for now...
extern int unique_signal; // for use in proving symbol stuff.

class Widget {
    public:
        virtual std::string message(void) = 0;
};