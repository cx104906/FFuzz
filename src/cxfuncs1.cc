#include <cstdlib>
#include <cstdio>
#include "cxconfig.h"
#include <string>

char cx_global_array1[MAP_SIZE + 4] = {0};
char cx_global_array2[MAP_SIZE2] = {0};

extern "C"
{
    void cxprintf1(const char *str)
    {
        printf("%s\n", str);
    }

    uint64_t cxfunc1()
    {
        uint64_t ret = 0;
        const char *env_val = std::getenv("CXENV1");
        if (!env_val || std::string(env_val) == "0")
        {
            ret = reinterpret_cast<uint64_t>(&cx_global_array1);
        }
        else
        {
            ret = std::stoull(env_val, nullptr, 10);
        }
        return ret;
    }
    uint64_t cxfunc2()
    {
        uint64_t ret = 0;
        const char *env_val = std::getenv("CXENV2");
        if (!env_val || std::string(env_val) == "0")
        {
            ret = reinterpret_cast<uint64_t>(&cx_global_array2);
        }
        else
        {
            ret = std::stoull(env_val, nullptr, 10);
        }
        return ret;
    }
}
