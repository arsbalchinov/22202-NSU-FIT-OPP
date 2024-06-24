#include <iostream>
#include <chrono>
#include <cmath>
#include <omp.h>

#define N 15360

using namespace std;
using chrono::high_resolution_clock;

double *initMatrix() {
    int i, j;
    double *A = new double[N * N];
#pragma omp parallel for private(i, j)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                A[i * N + j] = 2.0;
            }
            else {
                A[i * N + j] = 1.0;
            }
        }
    }
    return A;
}

double *initVector(double value) {
    double *res = new double[N];
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < N; i++) {
        res[i] = value;
    }
    return res;
}

void mult(const double *matrix, const double *vect, double *result) {
    int i, j;
#pragma omp parallel for private(i, j)
    for (i = 0; i < N; i++) {
        result[i] = 0.0;
        for (j = 0; j < N; j++) {
            result[i] += matrix[i * N + j] * vect[j];
        }
    }
}


void mult(double *a, const double tau) {
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < N; i++) {
        a[i] *= tau;
    }
}

void sub(double *a, const double *b) {
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < N; i++) {
        a[i] -= b[i];
    }
}

double Norm(const double *u) {
    double result = 0.0;
    int i;
#pragma omp parallel for reduction(+:result) private(i)
    for (i = 0; i < N; i++) {
        result += u[i] * u[i];
    }
    return sqrt(result);
}

struct Context {
    double *A;
    double *x;
    double *b;
    double *v;
    double tau;
    const double epsilon;
    double Norm_b;

    Context(const double epsilon) : epsilon(epsilon) {
        A = initMatrix();
        x = initVector(0.0);
        b = initVector(N + 1.0);
        v = initVector(0.0);
        Norm_b = Norm(b);
        tau = 0.0001;
    }

    ~Context() {
        delete[] A;
        delete[] x;
        delete[] b;
        delete[] v;
    }
};

bool isCloseEnough(Context &cont) {
    mult(cont.A, cont.x, cont.v);   //v = A*x^n
    sub(cont.v, cont.b);            //v = A*x^n - b
    double res = Norm(cont.v) / cont.Norm_b;
    return res < cont.epsilon;
}

void next(Context &cont) {
    mult(cont.A, cont.x, cont.v);   //v = A*x^n
    sub(cont.v, cont.b);            //v = A*x^n - b
    mult(cont.v, cont.tau);         //v = tau*(A*x^n - b)
    sub(cont.x, cont.v);            //x = x - tau*(A*x^n - b)
}

int main(int argc, char *argv[]) {
    const double epsilon = pow(10, -9);
    Context cont = Context(epsilon);
    high_resolution_clock::time_point begin = high_resolution_clock::now();
    while (!isCloseEnough(cont)) {
        next(cont);
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    cout << "Time diff = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" <<endl;
}
