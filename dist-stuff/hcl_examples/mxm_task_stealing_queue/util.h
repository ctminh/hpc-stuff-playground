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

char *getHostIB_IPAddr(){

    char *ret;
    int fd;
    struct ifreq ifr;

    fd = socket(AF_INET, SOCK_DGRAM, 0);

    // choose IPv4 type to get
    ifr.ifr_addr.sa_family = AF_INET;

    // get IP address attached to wlp61s0
    std::strncpy(ifr.ifr_name, "wlp61s0", IFNAMSIZ-1);
    ioctl(fd, SIOCGIFADDR, &ifr);
    close(fd);

    // to get the ip address
    ret = inet_ntoa(((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr);

    return ret;
}

#endif // UTIL_H