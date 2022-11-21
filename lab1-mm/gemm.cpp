#include <ctime>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <cilk/cilk.h>

// #define NAIVE
// #define LOOP_ORDER
// #define MEM
// #define COEFF
// #define CILK_LO
#define TILING
// #define CILK_DAC

#define MEASURE(__ret_ptr, __func, ...)           \
    ((clock_gettime(CLOCK_MONOTONIC, &start), \
      *(__ret_ptr) = __func(__VA_ARGS__),         \
      clock_gettime(CLOCK_MONOTONIC, &end)),  \
     (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec))

#define MEASURE_VOID(__func, ...)                 \
    ((clock_gettime(CLOCK_MONOTONIC, &start), \
      __func(__VA_ARGS__),                        \
      clock_gettime(CLOCK_MONOTONIC, &end)),  \
     (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec))

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define ABS(x) (((x) >= 0.0) ? (x) : -(x))
#define ISPOWER(x) ((x) & (-(x))) == (x)

int s = 128;
int DAC_THOLD = 32;

void RandomFill(struct drand48_data * buf_p, double *d, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        drand48_r(buf_p, d + i);
        d[i] = 2 * d[i] - 1.0;
    }
}

/* start of helper functions */
inline void mm_base(int m, int n, int k, double alpha,
            const double *A, const double *B, double *C, 
            int lda, int ldb, int ldc) {
    // "loop order" for C += alpha * A * B
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                C[i + j * ldc] += alpha * A[i + p * lda] * B[p + j * ldb];
            }
        }
    }
}

void mm_dac(int m, int n, int k, double alpha,
            const double *A, const double *B, double *C, 
            int ldA, int ldB, int ldC) {
    assert(ISPOWER(m) && ISPOWER(n) && ISPOWER(k));
    if (MIN(m, n) <= DAC_THOLD || k <= DAC_THOLD) {
        mm_base(m, n, k, alpha, A, B, C, ldA, ldB, ldC);
    } else {
        int& R_A = m; int& R_B = k; int& R_C = m;
        int& C_A = k; int& C_B = n; int& C_C = n; 
#define X(M, r, c) (M + r * (R_ ## M / 2) + c * (ld ## M) * (C_ ## M / 2))
        cilk_spawn mm_dac(m/2, n/2, k/2, alpha, X(A,0,0), X(B,0,0), X(C,0,0), ldA, ldB, ldC);
        cilk_spawn mm_dac(m/2, n/2, k/2, alpha, X(A,0,0), X(B,0,1), X(C,0,1), ldA, ldB, ldC);
        cilk_spawn mm_dac(m/2, n/2, k/2, alpha, X(A,1,0), X(B,0,0), X(C,1,0), ldA, ldB, ldC);
                   mm_dac(m/2, n/2, k/2, alpha, X(A,1,0), X(B,0,1), X(C,1,1), ldA, ldB, ldC);
        cilk_sync;
        cilk_spawn mm_dac(m/2, n/2, k/2, alpha, X(A,0,1), X(B,1,0), X(C,0,0), ldA, ldB, ldC);
        cilk_spawn mm_dac(m/2, n/2, k/2, alpha, X(A,0,1), X(B,1,1), X(C,0,1), ldA, ldB, ldC);
        cilk_spawn mm_dac(m/2, n/2, k/2, alpha, X(A,1,1), X(B,1,0), X(C,1,0), ldA, ldB, ldC);
                   mm_dac(m/2, n/2, k/2, alpha, X(A,1,1), X(B,1,1), X(C,1,1), ldA, ldB, ldC);
        cilk_sync;
    }
}

inline void mm_tiling(int m, int n, int k, double alpha,
            const double *A, const double *B, double *C, 
            int lda, int ldb, int ldc) {
    cilk_for (int jh = 0; jh < n; jh += s) {
        int J = MIN(n-jh, s);
        cilk_for (int ih = 0; ih < m; ih += s) {
            int I = MIN(m-ih, s);
            for (int ph = 0; ph < k; ph += s) {
                int P = MIN(k-ph, s);
                for (int jl = 0; jl < J; ++jl) {
                    int j = jh + jl;
                    for (int pl = 0; pl < P; ++pl) {
                        int p = ph + pl;
                        for (int il = 0; il < I; ++il) {
                            int i = ih + il;
                            C[i + j * ldc] += alpha * A[i + p * lda] * B[p + j * ldb];
                        }
                    }
                }
            }
        }
    }
}

void naive_gemm(int m, int n, int k, 
                const double * A, const double * B, double * C, 
                double alpha, double beta, 
                int lda, int ldb, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i + j * ldc] *= beta;
            for (int p = 0; p < k; p++) {
                C[i + j * ldc] += alpha * A[i + p * lda] * B[p + j * ldb];
            }
        }
    }
}
/* end of helper functions */

void student_gemm(int m, int n, int k, 
                  const double * A, const double * B, double * C, 
                  double alpha, double beta, 
                  int lda, int ldb, int ldc) {
    #ifdef NAIVE
    naive_gemm(m, n, k, A, B, C, alpha, beta, lda, ldb, ldc);
    #endif

    #ifdef LOOP_ORDER
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[i + j * ldc] *= beta;
        }
    }
    mm_base(m, n, k, alpha, A, B, C, lda, ldb, ldc);
    #endif

    #ifdef MEM
    double *AB = (double *)aligned_alloc(64, m * n * sizeof(double));
    memset(AB, 0, sizeof(double)*m*n);
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                AB[i + j * ldc] += A[i + p * lda] * B[p + j * ldb];
            }
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[i + j * ldc] = beta * C[i + j * ldc] + alpha * AB[i + j * ldc];
        }
    }
    free(AB);
    #endif

    #ifdef COEFF
    double ratio = beta / alpha;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[i + j * ldc] *= ratio;
        }
    }
    for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                C[i + j * ldc] += A[i + p * lda] * B[p + j * ldb];
            }
        }
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[i + j * ldc] *= alpha;
        }
    }
    #endif

    #ifdef CILK_LO
    cilk_for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[i + j * ldc] *= beta;
        }
    }
    cilk_for (int j = 0; j < n; j++) {
        for (int p = 0; p < k; p++) {
            for (int i = 0; i < m; i++) {
                C[i + j * ldc] += alpha * A[i + p * lda] * B[p + j * ldb];
            }
        }
    }
    #endif

    #ifdef TILING
    cilk_for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[i + j * ldc] *= beta;
        }
    }
    mm_tiling(m, n, k, alpha, A, B, C, lda, ldb, ldc);
    #endif

    #ifdef CILK_DAC
    cilk_for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[i + j * ldc] *= beta;
        }
    }
    if (ISPOWER(m) && ISPOWER(n) && ISPOWER(k)) {
        mm_dac(m, n, k, alpha, A, B, C, lda, ldb, ldc);
    } else {
        mm_tiling(m, n, k, alpha, A, B, C, lda, ldb, ldc);
    }
    #endif
}

void mm_test(int m, int n, int k)
{
    int lda = m;
    int ldb = k;
    int ldc = m;

    double *A = (double *)aligned_alloc(64, m * k * sizeof(double));
    double *B = (double *)aligned_alloc(64, k * n * sizeof(double));
    double *C = (double *)aligned_alloc(64, m * n * sizeof(double));
    double *C_ans = (double *)aligned_alloc(64, m * n * sizeof(double));
    
    struct drand48_data buffer;
    srand48_r(time(NULL), &buffer);

    RandomFill(&buffer, A, m * k);
    RandomFill(&buffer, B, k * n);
    RandomFill(&buffer, C, m * n);

    memcpy(C_ans, C, sizeof(double) * m * n);

    double alpha, beta;
    
    drand48_r(&buffer, &alpha);
    drand48_r(&buffer, &beta);
    alpha = 2 * alpha - 1.0;
    beta = 2 * beta - 1.0;

    /* test performance */

    const int TRIAL = 5;
    struct timespec start, end;
    double t_min = __DBL_MAX__;

    for (int i = 0; i < TRIAL; i++) {
        double t = MEASURE_VOID(student_gemm, m, n, k, A, B, C, alpha, beta, lda, ldb, ldc);

        t_min = MIN(t, t_min);
    }

    printf("minimal time spent: \e[1;36m\e[1m%.4f\e[0m ms\n", t_min * 1000);
    fflush(stdout);

    /* test correctness */

    memcpy(C, C_ans, sizeof(double) * m * n);
    student_gemm(m, n, k, A, B, C, alpha, beta, lda, ldb, ldc);
    naive_gemm(m, n, k, A, B, C_ans, alpha, beta, lda, ldb, ldc);

    double max_err = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i + j * ldc;
            double err = ABS(C_ans[idx] - C[idx]);
            max_err = MAX(err, max_err);
        }
    }
    const double threshold = 1e-7;
    const char * judge_s = (max_err < threshold) ? "\e[1;32m\e[1mcorrect\e[0m" : "\e[1;31m\e[1mwrong\e[0m";

    printf("result: %s (err = %e)\n", judge_s, max_err);

    free(A);
    free(B);
    free(C);
    free(C_ans);
}

int main(int argc, const char * argv[]) {
    if (argc != 4 && argc != 5) {
        printf("Test usage: ./test m n k [s]\n");
        exit(-1);
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    if (argc == 5) {
        DAC_THOLD = atoi(argv[4]);
    }

    printf("input: %d x %d x %d\n", m, n, k);
    fflush(stdout);

    mm_test(m, n, k);
}
