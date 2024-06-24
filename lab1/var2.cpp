                sum += matrix[i * size + j] * vect[j - offset];
            }
            result[i] += sum;
        }
        shiftData(vect);
        currRank = (currRank + num_of_proc - 1) % num_of_proc;
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
    double *multRes;
    double *subRes;
    const int size;
    const double epsilon;

    Context(const int n, const double epsilon) : size(n), epsilon(epsilon) {
        A = new double[size * dataLengths[proc_rank]];
        x = new double[dataLengths[0]];
        b = new double[dataLengths[0]];
        tau = 0.0001;
        multRes = new double[dataLengths[0]];
        subRes = new double[dataLengths[0]];
        for (int i = 0; i < dataLengths[proc_rank]; i++) {
            x[i] = 0;
            b[i] = size + 1;
        }
        for (int i = 0; i < dataLengths[proc_rank]; i++) {
            for (int j = 0; j < size; j++) {
                if (dataStarts[proc_rank] + i == j) {
                    A[i * size + j] = 2.0;
                } else {
                    A[i * size + j] = 1.0;
                }
            }

        }
    }
    ~Context() {
        delete[] A;
        delete[] x;
        delete[] b;
        delete[] multRes;
        delete[] subRes;
    }
};
bool isCloseEnough(Context &cont) {
    mult(cont.A, cont.x, cont.multRes, cont.size);
    sub(cont.multRes, cont.b, cont.subRes, dataLengths[proc_rank]);
    double res = Norm(cont.subRes) / Norm(cont.b);
    return res < cont.epsilon;
}
void next(Context &cont) {
    mult(cont.A, cont.x, cont.multRes, cont.size);
    sub(cont.multRes, cont.b, cont.subRes, dataLengths[proc_rank]);
    mult(cont.subRes, dataLengths[proc_rank], cont.tau);
    sub(cont.x, cont.subRes, cont.x, dataLengths[proc_rank]);
}
void InitRowsPerProcess(int n) {
    dataStarts = new int[num_of_proc];
    dataLengths = new int[num_of_proc];
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

    InitRowsPerProcess(N);
    Context cont = Context(N, epsilon);

    begin = MPI_Wtime();
    while (!isCloseEnough(cont)) {
        next(cont);
    }
    end = MPI_Wtime();

    cout << "Time diff = " << (end - begin) << "[s]" <<endl;
    delete[] dataStarts;
    delete[] dataLengths;
    MPI_Finalize();
}
