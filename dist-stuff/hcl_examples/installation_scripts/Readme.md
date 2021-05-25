## Installation Script for compiling HCL and it dependencies
The script should be adapted on your side with path, directories, ... Here, they are tested with my account on CoolMUC2 and BEAST system (https://www.lrz.de/presse/ereignisse/2020-11-06_BEAST/) at LRZ.

### 1. Install Spack and add SDS repository

**Note:** for simple we could use Intel OneAPI 2021 as the main compilers for our compilation in further. So, remember to load intel oneapi on CoolMUC2 before adding mochi-packages.

First, you will need to install Spack as explained here: https://spack.readthedocs.io/en/latest/getting_started.html

Once Spack is installed and available in your path, clone the following git reporitory and add it as a Spack namespace.
```Bash
git clone https://github.com/mochi-hpc/mochi-spack-packages.git

spack repo add mochi-spack-packages
```
Now, we can check the information of the SDS packages that would be needed for HCL library. More info could be found here: https://mochi.readthedocs.io/en/latest/installing.html.

```Bash
spack info mochi-thallium
```
```
> CMakePackage:   mochi-thallium
>
> Description:
>    A Mochi C++14 library wrapping Margo, Mercury, and Argobots and
>    providing an object-oriented way to use these libraries.
>
> Homepage: https://github.com/mochi-hpc/mochi-thallium
> ...
```

**Note:** for ful functionality of HCL library with RPC over RDMA, we need to install mochi-thallium packages and dependencies. However, to enable rdma with verbs on CoolMUC2 or BEAST system, we should tell Spack using some built-in packages, e.g., `rdma-core`.

To this end, we coudld do as follow: create a config file in our spack workspace. As default, it should be here: `~./spack/`. First, check the installed rdma-core package on the system.
```Bash
rpm -qa | grep rdma-core
```

The result could be: $rdma-core-47mlnx1-1.47329.x86_64$ on CoolMUC2. Then, add this info for the Spack configuration in packages.yaml file by Vim or whatever:

``` Bash
vim ~/.spack/packages.yaml
```
```
packages:
        rdma-core:
                externals:
                - spec: rdma-core@47mlnx1-1.47329.x86_64
                  prefix: /usr
                buildable: false
                version: []
                target: []
                providers: {}
                compiler: []
```
After `@` could be the version or whatever we want, to indicate the version.

### 2. Install mochi-thallium with enabling Ib-Verbs
Once everything is ready, we could use Spack to install mochi-thallium first.
```Bash
spack install mochi-thallium@0.7 %oneapi@2021.1 ^libfabric  fabrics=tcp,verbs,rxm,mrail
```
Where, `@0.7` is the version of thallium, `@0.8` is the newest one but `@0.7` should be fine on CoolMUC2 or BEAST. `%oneapi@2021.1` is the compiler we would tell Spack that it should install thallium with. `^libfabric` is one of the main dependencies in Thallium. Importantly, we should tell Spack that it should compile `libfabric` with the following protocols, e.g., tcp, verbs, rxm, mrail.

Finally, if everything works, we will have the installed thallium in Spack directories.

### 3. Compile HCL
**Note:** before compiling HCL, to make it works on CoolMUC2 or BEAST, we should change some configurations in the src-file `/include/hcl/common/configuration_manager.h`. At line #62, we should change it into:
```C
    TCP_CONF("ofi+sockets"), VERBS_CONF("ofi+verbs"), VERBS_DOMAIN("ofi_rxm")
```
Then, we can compile HCl with the loaded dependencies above as one of config-scripts in this folder.