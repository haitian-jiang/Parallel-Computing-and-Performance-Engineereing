#include <omp.h>
#include "utils.hpp"

int lcs_basic(const char * _A, const char * _B, int M, int N) {
    int (*dp)[2] = (int(*)[2])calloc(N * 2, sizeof(int));  // N * 2 is better for cache if N is large
    bool bi;
    for (int i = 0; i < M; ++i) {
        bi = i & 1;
        for (int j = 0; j < N; ++j) {
            int up = (i > 0) ? dp[j][1 - bi] : 0;
            int left = (j > 0) ? dp[j - 1][bi] : 0;
            int upleft = (i > 0 && j > 0) ? dp[j - 1][1 - bi] : 0;
            if (_A[i] == _B[j]) {
                dp[j][bi] = upleft + 1;
            } else {
                dp[j][bi] = max(up, left);
            }
        }
    }
    int res = dp[N - 1][bi];
    free(dp);
    return res;
}

int lcs_ad(const char * _A, const char * _B, int M, int N) {
    int* dp = (int*)calloc(M + N - 1, sizeof(int));
    for (int s = 0; s < M + N - 1; ++s) {  // s = i + j
        int start = (s < M) ? (M - s -1) : (s - M + 1);
        int end = (s < N) ? (M + s + 1) : (N * 2 + M - s - 1);
        for (int t = start; t < end; t += 2) {  // t = j - i + M - 1
            int i = (s + M - t - 1) / 2;
            int j = (s + t + 1 - M) / 2;
            if (_A[i] == _B[j]) {
                ++dp[t];
            } else {
                dp[t] = max(dp[t-1], dp[t+1]);
            }
        }
    }
    int res = dp[N-1];
    free(dp);
    return res;
}

int lcs_ad_parallel(const char * _A, const char * _B, int M, int N) {
    int* dp = (int*)calloc(M + N - 1, sizeof(int));
    for (int s = 0; s < M + N - 1; ++s) {  // s = i + j
        int start = (s < M) ? (M - s -1) : (s - M + 1);
        int end = (s < N) ? (M + s + 1) : (N * 2 + M - s - 1);
        int t;
        // #pragma omp parallel for num_threads(10) private(t) schedule(static, 16)
        #pragma omp parallel for
        for (t = start; t < end; t += 2) {  // t = j - i + M - 1
            int i = (s + M - t - 1) / 2;
            int j = (s + t + 1 - M) / 2;
            if (_A[i] == _B[j]) {
                ++dp[t];
            } else {
                dp[t] = max(dp[t-1], dp[t+1]);
            }
        }
    }
    int res = dp[N-1];
    free(dp);
    return res;
}

