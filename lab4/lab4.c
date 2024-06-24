#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>

#define hx2 hx*hx
#define hy2 hy*hy
#define hz2 hz*hz

const double epsilon = 1e-8;
const double a = 1e5;

int Nx = 256, Ny = 256, Nz = 384;
double Dx = 2.0, Dy = 2.0, Dz = 2.0;
double Ox = -1.0, Oy = -1.0, Oz = -1.0;
double hx, hy, hz;
double coeff;

double phi(double x, double y, double z){
    return x*x + y*y + z*z;
}
double rho(double x, double y, double z){
    return 6.0 - a * phi(x, y, z);
}
double max(double a, double b){
    return a > b ? a : b;
}

MPI_Comm CreateLinearComm() {
    int commsize;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);

    int ndims = 1;
    int dims[1] = {commsize};
    int periods[1] = {false};
    bool reorder = true;

    MPI_Comm linear_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &linear_comm);

    return linear_comm;
}

void splitGrid(int num_of_proc, int procrank) {
    Nz /= num_of_proc;
    Dz /= num_of_proc;
    Oz += procrank * Dz;
    hx = Dx / (Nx - 1);
    hy = Dy / (Ny - 1);
    hz = Dz / (Nz - 1);
    coeff = 1 / (2 / (hx2) + 2 / (hy2) + 2 / (hz2) + a);
}

void initGrid(double (*grid)[Nx][Ny], bool neighbor_below, bool neighbor_above) {
    for (int k = 0; k < Nz; k++){
        for (int i = 0; i < Nx; i++){
            for (int j = 0; j < Ny; j++){
                double x = Ox + i * hx;
                double y = Oy + j * hy;
                double z = Oz + k * hz;

                if (i == 0 || i == Nx-1 || j == 0 || j == Ny-1 || k == 0 && !neighbor_below || k == Nz-1 && !neighbor_above){
                    grid[k][i][j] = phi(x, y, z);
                }
                else{
                    grid[k][i][j] = 0.0;
                }
            }
        }
    }
}

void initBuff(double (*buff)[Nx][Ny]){
    for (int i = 1; i < Nx - 1; i++){
        for (int j = 1; j < Ny - 1; j++){
            buff[0][i][j] = 0.0;
            buff[1][i][j] = 0.0;
        }
    }
}

void exchangeBoundaries(double (*grid)[Nx][Ny], double (*buff)[Nx][Ny], MPI_Comm comm, MPI_Request* request){
    int count = Nx * Ny;
    int sendcounts[2] = {count, count};
    int recvcounts[2] ={count, count};
    int sdispls[2] = {0, (Nz - 1) * count};
    int rdispls[2] = {0, count};

    MPI_Ineighbor_alltoallv(grid, sendcounts, sdispls, MPI_DOUBLE, buff, recvcounts, rdispls, MPI_DOUBLE, comm, request);
}

void updateInnerValues(double (*curr)[Nx][Ny], double (*next)[Nx][Ny], double *delta){
    for (int k = 1; k < Nz - 1; k++){
        for (int i = 1; i < Nx - 1; i++){
            for (int j = 1; j < Ny - 1; j++){
                double x = Ox + i * hx;
                double y = Oy + j * hy;
                double z = Oz + k * hz;

                next[k][i][j] = coeff * (
                        (curr[k][i + 1][j] + curr[k][i - 1][j]) / (hx2) +
                        (curr[k][i][j + 1] + curr[k][i][j - 1]) / (hy2) +
                        (curr[k + 1][i][j] + curr[k - 1][i][j]) / (hz2) -
                        rho(x, y, z));

                *delta = max(*delta, fabs(next[k][i][j] - curr[k][i][j]));
            }
        }
    }
}

void updateBoundaryValues(double (*curr)[Nx][Ny], double (*next)[Nx][Ny], double (*buff)[Nx][Ny], double *delta, bool neighbor_below, bool neighbor_above) {
    for (int i = 1; i < Nx - 1; i++){
        for (int j = 1; j < Ny - 1; j++){
            double x = Ox + i * hx;
            double y = Oy + j * hy;

            if (neighbor_above){
                int k = Nz - 1;
                double z = Oz + k * hz;
                next[k][i][j] = coeff * (
                        (curr[k][i + 1][j] + curr[k][i - 1][j]) / (hx2) +
                        (curr[k][i][j + 1] + curr[k][i][j - 1]) / (hy2) +
                        (buff[1][i][j] + curr[k - 1][i][j]) / (hz2) -
                        rho(x, y, z));

                *delta = max(*delta, fabs(next[k][i][j] - curr[k][i][j]));
            }
            if (neighbor_below){
                int k = 0;
                double z = Oz + k * hz;
                next[k][i][j] = coeff * (
                     (curr[k][i + 1][j] + curr[k][i - 1][j]) / (hx * hx) +
                     (curr[k][i][j + 1] + curr[k][i][j - 1]) / (hy * hy) +
                     (curr[k + 1][i][j] + buff[0][i][j]) / (hz * hz) -
                     rho(x, y, z));

                *delta = max(*delta, fabs(next[k][i][j] - curr[k][i][j]));
            }
        }
    }
}

void solveJacobi() {
    MPI_Comm linear_comm = CreateLinearComm();

    int num_of_proc, procrank;
    MPI_Comm_size(linear_comm, &num_of_proc);
    MPI_Comm_rank(linear_comm, &procrank);

    splitGrid(num_of_proc, procrank);

    bool neighbor_below = procrank != 0;
    bool neighbor_above = procrank != num_of_proc - 1;

    double(*curr)[Nx][Ny] = malloc(sizeof(double[Nz][Nx][Ny]));
    initGrid(curr, neighbor_below, neighbor_above);

    double(*buff)[Nx][Ny] = malloc(sizeof(double[2][Nx][Ny]));
    initBuff(buff);

    double delta;
    double(*next)[Nx][Ny];

    do {
        next = malloc(sizeof(double[Nz][Nx][Ny]));
        delta = 0;

        updateBoundaryValues(curr, next, buff, &delta, neighbor_below, neighbor_above);

        MPI_Request exchange_request;
        exchangeBoundaries(curr, buff, linear_comm, &exchange_request);

        updateInnerValues(curr, next, &delta);

        MPI_Wait(&exchange_request, MPI_STATUS_IGNORE);

        MPI_Allreduce(MPI_IN_PLACE, &delta, 1, MPI_DOUBLE, MPI_MAX, linear_comm);

        free(curr);
        curr = next;
    } while (delta >= epsilon);

    free(curr);
    free(buff);

    MPI_Comm_free(&linear_comm);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    double begin = MPI_Wtime();

    solveJacobi();

    double end = MPI_Wtime();

    printf("Time taken: %lf [s]\n", end-begin);

    MPI_Finalize();
    return 0;
}
