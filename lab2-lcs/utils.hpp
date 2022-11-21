#include <string>
#include <cstdlib>
#include <ctime>

#define UTILS_DEBUG

static struct timespec __start, __end;

#define MEASURE(__ret_ptr, __func, ...) \
    ((clock_gettime(CLOCK_MONOTONIC, &__start), \
    *(__ret_ptr) = __func(__VA_ARGS__),       \
    clock_gettime(CLOCK_MONOTONIC, & __end)),  \
    (__end.tv_sec - __start.tv_sec) + 1e-9 * (__end.tv_nsec - __start.tv_nsec))\

#define MEASURE_NO_RETURN(__func, ...) \
    ((clock_gettime(CLOCK_MONOTONIC, &__start), \
    __func(__VA_ARGS__),       \
    clock_gettime(CLOCK_MONOTONIC, & __end)),  \
    (__end.tv_sec - __start.tv_sec) + 1e-9 * (__end.tv_nsec - __start.tv_nsec)) \

int max(const int a, const int b);
int min(const int a, const int b);

int naive_lcs_2d(const std::string A, const std::string B);