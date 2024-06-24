#include <iostream>
#include <cmath>
#include <mpi.h>
#define N 17280
using namespace std;

int num_of_proc;
int proc_rank;
double *buffer;
int *dataStarts;
int *dataLengths;

double Norm(const double *u, const int size) {
    double res = 0;
    for (int i = 0; i < size; i++) {
        res += u[i] * u[i];
    }
    return sqrt(res);
}

void mult(const double *matrix, const double *vect, double *result, int size) {
    for (int i = 0; i < dataLengths[proc_rank]; i++) {
        double sum = 0;
        for (int j = 0; j < size; j++) {
            sum += matrix[i * size + j] * vect[j];
        }
        result[i] = sum;
    }
}

void mult(double *a, const int size, const double tau) {
    for (int i = 0; i < size; i++) {
        a[i] *= tau;
    }
}
void sub(const double *a, const double *b, double *c, const int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] - b[i];
    }
}
struct Context {
    double *A;
    double *x;
    double *b;
    double tau;
    double *vMult;    //A*x^n
    double *vSub;     //A*x^n - b
    const int size;
    const double epsilon;
    Context(const int n, const double epsilon) : size(n), epsilon(epsilon) {
        A = new double[size * dataLengths[proc_rank]];
        x = new double[size];
        b = new double[size];
        tau = 0.0001;
        vMult = new double[dataLengths[proc_rank]];
        vSub = new double[dataLengths[proc_rank]];

        for (int i = 0; i < size; i++) {
            x[i] = 0;
            b[i] = size + 1;
        }
        for (int i = 0; i < dataLengths[proc_rank]; i++) {
            for (int j = 0; j < size; j++) {
                if (dataStarts[proc_rank] + i == j) {
                    A[i * size + j] = 2.0;
                }
                else {
                    A[i * size + j] = 1.0;
                }
            }
        }
    }
    ~Context() {
        delete[] A;
        delete[] x;
        delete[] b;
        delete[] vMult;
        delete[] vSub;
    }
};
void gatherVector(double *from, double *to) {
    MPI_Allgatherv(from, dataLengths[proc_rank],  MPI_DOUBLE, to, dataLengths, dataStarts, MPI_DOUBLE, MPI_COMM_WORLD);
}

bool isCloseEnough(Context &cont) {
    mult(cont.A, cont.x, cont.vMult, cont.size);
    sub(cont.vMult, cont.b, cont.vSub, dataLengths[proc_rank]);
    gatherVector(cont.vSub, buffer);
    double res = Norm(buffer, cont.size) / Norm(cont.b, cont.size);
    return res < cont.epsilon;
}

void next(Context &cont) {
    mult(cont.A, cont.x, cont.vMult, cont.size);
    sub(cont.vMult, cont.b, cont.vSub, dataLengths[proc_rank]);
    mult(cont.vSub, dataLengths[proc_rank], cont.tau);
    sub(cont.x, cont.vSub, cont.vMult, dataLengths[proc_rank]);
    gatherVector(cont.vMult, cont.x);
}
void InitRowsPerProcess(int n) {
    dataStarts = new int[n];
    dataLengths = new int[n];
    int perProcess = n / num_of_proc;
    int ost = n % num_of_proc;
    int start = 0;
    for (int i = 0; i < ost; i++) {
        dataStarts[i] = start;
        dataLengths[i] = perProcess + 1;
        start += perProcess + 1;
    }
    for (int i = ost; i < num_of_proc; i++) {
        dataStarts[i] = start;
        dataLengths[i] = perProcess;
        start += perProcess;
    }
}

int main(int argc, char *argv[]) {
    double begin, end;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    const double epsilon = pow(10, -9);

    InitRowsPerProcess(N);  //Разрезаем матрицу
    Context cont = Context(N, epsilon);
    buffer = new double[N];

    begin = MPI_Wtime();
    while (!isCloseEnough(cont)) {
        next(cont);
    }
    end = MPI_Wtime();

    cout << "Time diff = " << (end - begin) << "[s]" <<endl;
    delete[] dataStarts;
    delete[] dataLengths;
    delete[] buffer;
    MPI_Finalize();
}
