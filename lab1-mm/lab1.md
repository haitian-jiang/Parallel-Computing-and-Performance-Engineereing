# Lab 1：矩阵乘法

## 问题描述

输入三个矩阵
$$
A\in\R^{m\times k},B\in\R^{k\times n},C\in\R^{m\times n}
$$
和$\alpha,\beta\in\R$，请计算$\alpha\cdot AB+\beta\cdot C$，并且计算结果保存在输入矩阵$C$。具体来说，请编写Ｃ语言程序完成矩阵乘法的接口函数。

```C
void student_gemm(int m, int n, int k, const double * A, const double * B, double * C, double alpha, double beta, int lda, int ldb, int ldc);
```

输入矩阵的格式规定是列主序（column major），即第一个元素 A[0, 0] 下一个元素为 A[1, 0] 而非 A[0, 1]。你可以透过 A[i + j * m] 方式索引一个 m x n 矩阵Ａ的第 i 行、第 j 列的元素 Aij。

参数 lda, ldb, ldc 表示矩阵 A, B, C 的列偏移（Leading Dimension of A, B, C）。当你的输入矩阵是大矩阵中的子矩阵，如果继续使用 A[i + j * m] 方式索引将会出错，因此在列主序下读取元素 Aij 的通用方式为A[i + j * lda]。你或许需要思考一下调用上述接口时 lda, ldb, ldc 的初始值。

## 框架代码

```cpp
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <ctime>

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

void RandomFill(struct drand48_data * buf_p, double *d, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        drand48_r(buf_p, d + i);
        d[i] = 2 * d[i] - 1.0;
    }
}

void student_gemm(int m, int n, int k, const double * A, const double * B, double * C, double alpha, double beta, int lda, int ldb, int ldc)
{
    /* TODO */
}

void naive_gemm(int m, int n, int k, const double * A, const double * B, double * C, double alpha, double beta, int lda, int ldb, int ldc)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i + j * ldc] *= beta;
            for (int p = 0; p < k; p++) {
                C[i + j * ldc] += alpha * A[i + p * lda] * B[p + j * ldb];
            }
        }
    }
    return;
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

    printf("minimal time spent: %.4f ms\n", t_min * 1000);
    fflush(stdout);

    /* test correctness */

    memcpy(C, C_ans, sizeof(double) * m * n);
    student_gemm(m, n, k, A, B, C, alpha, beta, lda, ldb, ldc);
    naive_gemm(m, n, k, A, B, C_ans, alpha, beta, lda, ldb, ldc);

    double max_err = __DBL_MIN__;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i + j * ldc;
            double err = ABS(C_ans[idx] - C[idx]);
            max_err = MAX(err, max_err);
        }
    }
    const double threshold = 1e-7;
    const char * judge_s = (max_err < threshold) ? "correct" : "wrong";

    printf("result: %s (err = %e)\n", judge_s, max_err);

    free(A);
    free(B);
    free(C);
    free(C_ans);
}

int main(int argc, const char * argv[])
{
    if (argc != 4)
    {
        printf("Test usage: ./test m n k\n");
        exit(-1);
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    printf("input: %d x %d x %d\n", m, n, k);
    fflush(stdout);

    mm_test(m, n, k);
}

```

### 助教批改原则

首先，你必须答案正确，才能获得基本分数。

本次实验的重点是性能优化，你的矩阵乘法函数计算速度越快越好。请注意，矩阵乘法的性能表现攸关你的成绩，这意味着抄袭框架代码中用来检查正确性的 naive_gemm 函数并没有意义。

另外，请勿修改框架代码，同学们唯一的任务是完成 student_gemm 函数。如果框架代码存在问题，请及时反馈，助教将会尽快修正文档。

## 提交方法

实验代码必须提交到校内服务器。进入校内网后（可能需要復旦大学校园VPN），使用命令行或SSH相关工具登入服务器。命令行SSH指令：

```
ssh [your-student-ID]@10.192.9.250
```

所有人的密码均为123。

请将代码与实验报告一起放在相同的文件夹底下，文件夹命名格式为 [student-id]_lab_1。你可以透过 scp 指令上传至自己帐户的目录/home/[your-student-id] 。关于具体的指令使用方法，请参考 scp 指令相关资料。

**提交截止时间：2022.10.26 23:59**
