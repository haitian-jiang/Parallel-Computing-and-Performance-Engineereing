#include <cstdio>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include "utils.hpp"
#include <set>
#include <cassert>

#define CORRECT "\e[1;32m\e[1mcorrect\e[0m"
#define WRONG "\e[1;31m\e[1mwrong\e[0m"


int lcs_basic(const char * _A, const char * _B, int M, int N);
int lcs_ad(const char * _A, const char * _B, int M, int N); // LCS with anti-diagonal method
int lcs_ad_parallel(const char * _A, const char * _B, int _M, int _N);

int main(int argc, char *argv[])
{
    assert(argc >= 2);

    std::string _input = argv[1];

    std::string A, B;

    std::ifstream input_file(_input);
    input_file >> A >> B;
    input_file.close();

    const int M = A.length();
    const int N = B.length();
    const char * A_str = A.c_str();
    const char * B_str = B.c_str();

    printf("------------------------------------------\n");
    printf("M: %d\nN: %d\n", M, N);

    int lcs_len;
    int answer_len = naive_lcs_2d(A, B);

    double tdiff;

    tdiff = MEASURE(&lcs_len, lcs_basic, A_str, B_str, M, N);
    printf("lcs_1d(%s): %.3f (ms)\n", (lcs_len == answer_len) ? CORRECT : WRONG, tdiff * 1e3);

    tdiff = MEASURE(&lcs_len, lcs_ad, A_str, B_str, M, N);
    printf("lcs_ad(%s): %.3f (ms)\n", (lcs_len == answer_len) ? CORRECT : WRONG, tdiff * 1e3);

    tdiff = MEASURE(&lcs_len, lcs_ad_parallel, A_str, B_str, M, N);
    printf("lcs_ad_parallel(%s): %.3f (ms)\n", (lcs_len == answer_len) ? CORRECT : WRONG, tdiff * 1e3);

    return 0;
}
