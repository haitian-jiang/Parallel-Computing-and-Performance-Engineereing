#include "utils.hpp"
#include <cstdlib>
#include <cassert>
#include <string>
#include <queue>
#include <cstdlib>
#include <stack>
#include <cstring>

int max(const int a, const int b)   
{
    return (a > b) ? a : b;
}

 int min(const int a, const int b)
{
    return (a > b) ? b : a;
}

int naive_lcs_2d_(const std::string A, const std::string B)
{
    const int M = A.length();
    const int N = B.length();

    int * dp = (int *)calloc(M * N, sizeof(int));

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int up = (i > 0) ? dp[(i - 1) * N + j] : 0;
            int left = (j > 0) ? dp[i * N + j - 1] : 0;
            int upleft = (i > 0 && j > 0) ? dp[(i - 1) * N + j - 1] : 0;

            if (A[i] == B[j])
                dp[i * N + j] = upleft + 1;
            else 
                dp[i * N + j] = max(up, left);
        }
    }

    int res = dp[M * N - 1];
    free(dp);

    return res;
}

int naive_lcs_2d(const std::string A, const std::string B)
{
    const int M = A.length();
    const int N = B.length();

    int* dp = (int*)calloc(N, sizeof(int));

    for (int i = 0; i < M; ++i) {
        int upleft = dp[0];
        if (A[i] == B[0])
            dp[0] = 1;

        for (int j = 1; j < N; ++j) {
            int up = dp[j], left = dp[j - 1];
            int cur;

            if (A[i] == B[j])
                cur = upleft + 1;

            else
                cur = (up > left) ? up : left;

            upleft = dp[j];
            dp[j] = cur;
        }
    }

    int res = dp[N - 1];
    free(dp);
    return res;
}
