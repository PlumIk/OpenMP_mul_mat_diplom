#include<algorithm> // for std::min()
#include<cstdio>    // for printf
#include<chrono>    // for time measurement
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void my_dgemm5(int n,int si,int sj,int sk, int num,const double *a,const double *b,double *c) {
    #pragma omp parallel for num_threads(num) collapse(2)
    for (int bi = 0; bi < n; bi += si)
        for (int bj = 0; bj < n; bj += sk)
            for (int bk = 0; bk < n; bk += sj)
                for (int i = bi; i < std::min(n,bi+si); i++)
                    for (int k = bj; k < std::min(n,bj+sj); k++)
                        for (int j = bk; j < std::min(n,bk+sk); j++)
                            c[i*n + j] += a[i*n + k] * b[k*n + j];
}

int main(int argc, char* argv[]) {
    int n = 1300;
    double *c = new double [n*n];
    double *a = new double [n*n];
    double *b = new double [n*n];
    int si = atoi(argv[1]);
    int sj = atoi(argv[2]);
    int sk = atoi(argv[3]);
    int num = atoi(argv[4]);
    auto t2 = std::chrono::high_resolution_clock::now();
    my_dgemm5(n, si, sj,sk, num, a,b,c);
    auto t3 = std::chrono::high_resolution_clock::now();
    double dt_blocked = 1.0e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    printf("Time:%lf\n",dt_blocked);
}
