#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <cmath>
#include <string>
#include <cstring>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>
#include <boost/stacktrace.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>

#ifndef TEST_ON_LAPTOP
#define TEST_ON_LAPTOP 0
#endif

/**
 * Getting IP_addr of IB devices
 * 
 * @param None
 * @return the IB-IP address corresponding to the interface name "ib0"
 * Depends on the system, it works on CoolMUC2 and BEAST system.
 */
char *getHostIB_IPAddr(){

    char *ret;
    int fd;
    struct ifreq ifr;

    fd = socket(AF_INET, SOCK_DGRAM, 0);

    // choose IPv4 type to get
    ifr.ifr_addr.sa_family = AF_INET;

#if TEST_ON_LAPTOP==1
    // get IP address attached to localhost
    // std::cout << "[DBG] getting local-addr..." << std::endl;
    std::strncpy(ifr.ifr_name, "lo", IFNAMSIZ-1);
#else
    // get IP address attached to ib0
    // std::cout << "[DBG] getting IB-addr..." << std::endl;
    std::strncpy(ifr.ifr_name, "ib0", IFNAMSIZ-1);
#endif
    
    ioctl(fd, SIOCGIFADDR, &ifr);
    close(fd);

    // to get the ip address
    ret = inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr);

    return ret;
}


/**
 * Tracing stack-call with Boost-StackTrace
 * 
 * @param None
 * @return print the current calls on stack.
 */


#endif // UTIL_H