#include <iostream>
#include <mpi.h>

#define M 256
#define N 256
#define K 256

using namespace std;

void initMatrix(double* matrix, int n, int m, double val) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i * n + j] = val;
        }
    }
}

void mult(int *matrixSizes, double *A, double *B, double *C, int *gridSizes, MPI_Comm comm) {
    double *submA = nullptr;
    double *submB = nullptr;
    double *submC = nullptr;
    int submSizes[2];

    int coords[2];
    int rank;

    int *sendcountsB = nullptr;
    int *displsScatterB = nullptr;

    int *recvcountsGatherC = nullptr;
    int *displsGatherC = nullptr;

    MPI_Datatype typeB, typeC, types[2];
    int blockLengths[2] = {1, 1};
    int periods[2] = {0, 0};
    int remainDims[2];

    MPI_Comm comm2d;
    MPI_Comm comm1d[2];

    MPI_Bcast(matrixSizes, 3, MPI_INT, 0, comm);
    MPI_Bcast(gridSizes, 2, MPI_INT, 0, comm);

    MPI_Cart_create(comm, 2, gridSizes, periods, false, &comm2d);
    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_coords(comm2d, rank, 2, coords);


    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            remainDims[j] = (i == j);
        }
        MPI_Cart_sub(comm2d, remainDims, &comm1d[i]);
    }

    submSizes[0] = matrixSizes[0] / gridSizes[0];
    submSizes[1] = matrixSizes[2] / gridSizes[1];

    submA = new double[submSizes[0] * matrixSizes[1]];
    submB = new double[matrixSizes[1] * submSizes[1]];
    submC = new double[submSizes[0] * submSizes[1]];

    if (rank == 0) {
        MPI_Type_vector(matrixSizes[1], submSizes[1], matrixSizes[2], MPI_DOUBLE, &types[0]);
        long int size;
        MPI_Type_extent(MPI_DOUBLE, &size);
        types[1] = MPI_UB;

        long int *displacements = new long int[2];
        displacements[0] = 0;
        displacements[1] = size * submSizes[1];

        MPI_Type_create_struct(2, blockLengths, displacements, types, &typeB);
        MPI_Type_commit(&typeB);

        displsScatterB = new int[gridSizes[1]];
        sendcountsB = new int[gridSizes[1]];

        for (int i = 0; i < gridSizes[1]; i++) {
            displsScatterB[i] = i;
            sendcountsB[i] = 1;
        }

        MPI_Type_vector(submSizes[0], submSizes[1], matrixSizes[2], MPI_DOUBLE, &typeC);
        MPI_Type_create_struct(2, blockLengths, displacements, types, &typeC);
        MPI_Type_commit(&typeC);

        displsGatherC = new int[gridSizes[0] * gridSizes[1]];
        recvcountsGatherC = new int[gridSizes[0] * gridSizes[1]];

        for (int i = 0; i < gridSizes[0]; i++) {
            for (int j = 0; j < gridSizes[1]; j++) {
                displsGatherC[i * gridSizes[1] + j] = (i * gridSizes[1] * submSizes[0] + j);
                recvcountsGatherC[i * gridSizes[1] + j] = 1;
            }
        }

        delete[] displacements;
    }

    if (coords[1] == 0) {
        MPI_Scatter(A, submSizes[0] * matrixSizes[1], MPI_DOUBLE, submA, submSizes[0] * matrixSizes[1], MPI_DOUBLE, 0, comm1d[0]);
    }

    if (coords[0] == 0) {
        MPI_Scatterv(B, sendcountsB, displsScatterB, typeB, submB, submSizes[1] * matrixSizes[1], MPI_DOUBLE, 0, comm1d[1]);
    }

    MPI_Bcast(submA, submSizes[0] * matrixSizes[1], MPI_DOUBLE, 0, comm1d[1]);
    MPI_Bcast(submB, submSizes[1] * matrixSizes[1], MPI_DOUBLE, 0, comm1d[0]);

    int m = submSizes[1];
    int n = matrixSizes[1];
    for (int i = 0; i < submSizes[0]; i++) {
        for (int j = 0; j < m; j++) {
            submC[i * m + j] = 0;
            for (int k = 0; k < n; k++) {
                submC[i * m + j] += submA[n * i + k] * submB[m * k + j];
            }
        }
    }

    MPI_Gatherv(submC, submSizes[0] * submSizes[1], MPI_DOUBLE, C, recvcountsGatherC, displsGatherC, typeC, 0, comm2d);

    delete[] submA;
    delete[] submB;
    delete[] submC;
    delete[] displsScatterB;
    delete[] sendcountsB;
    MPI_Comm_free(&comm2d);
    MPI_Comm_free(&comm1d[0]);
    MPI_Comm_free(&comm1d[1]);

    if (rank == 0) {
        delete[] recvcountsGatherC;
        delete[] displsGatherC;
        MPI_Type_free(&typeB);
        MPI_Type_free(&typeC);
        MPI_Type_free(&types[0]);
    }
}



int main(int argc, char *argv[]) {
    double begin, end;
    int size, rank;
    int matrixSizes[3];
    int gridSizes[2];
    double *A = nullptr;
    double *B = nullptr;
    double *C = nullptr;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3) {
        if (rank == 0) {
            cout << "Empty grid size"<<endl;
        }
        exit(1);
    }

    int p1 = atoi(argv[1]);
    int p2 = atoi(argv[2]);
    if (p1 == 0 || p2 == 0) {
        if (rank == 0) {
            cout << "Invalid grid size"<<endl;
        }
        exit(1);
    }

    if (size != p1 * p2) {
        if (rank == 0) {
            cout << "Wrong grid size"<<endl;
        }
        exit(1);
    }
    if (rank == 0) {
        matrixSizes[0] = M;
        matrixSizes[1] = N;
        matrixSizes[2] = K;

        gridSizes[0] = p1;
        gridSizes[1] = p2;

        A = new double[M * N];
        B = new double[N * K];
        C = new double[M * K];

        initMatrix(A, M, N, 1);
        initMatrix(B, N, K, 1);
        initMatrix(C, M, K, 0);
    }

    begin = MPI_Wtime();
    mult(matrixSizes, A, B, C, gridSizes, MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (rank == 0) {
        cout << "Time taken: " << end - begin << " [s]"<<endl;
        delete[] A;
        delete[] B;
        delete[] C;
    }

    MPI_Finalize();
    return 0;
}
