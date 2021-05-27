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

#ifndef TEST_ON_LAPTOP
#define TEST_ON_LAPTOP 0
#endif

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

#endif // UTIL_H