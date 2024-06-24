#include <iostream>
#include <unistd.h>
#include <pthread.h>
#include <mpi.h>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define EXECUTOR_FINISHED_WORK -1
#define SENDING_TASKS 656
#define SENDING_TASK_COUNT 787
#define NO_TASKS_TO_SHARE -565

#define ITERATION_COUNT 10
#define TASK_COUNT 2000
#define MIN_TASKS_TO_SHARE 2

using namespace std;

pthread_t threads[2];
pthread_mutex_t mutex;
int* tasks;

double summaryDisbalance = 0;
bool isFinishedExecution = false;

int num_of_proc;
int procrank;
int remainingTasks;
int executedTasks;
int additionalTasks;
double globalRes = 0;

void initTasks(int *taskSet, int taskCount, int iterCount) {
    for (int i = 0; i < taskCount; i++) {
        taskSet[i] = abs(procrank - (iterCount % num_of_proc));
    }
}

void executeTasks(const int* taskSet) {
    pthread_mutex_lock(&mutex);
    int startCount = remainingTasks;
    pthread_mutex_unlock(&mutex);

    for (int i = 0; i < startCount; i++) {
        pthread_mutex_lock(&mutex);
        if (startCount >= remainingTasks && i >= remainingTasks) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        int weight = taskSet[i];
        pthread_mutex_unlock(&mutex);

        for (int j = 0; j < weight; j++) {
            globalRes += sqrt(j);
        }

        executedTasks++;
    }
    remainingTasks = 0;
}

void* executor(void* args) {
    tasks = new int[TASK_COUNT];
    double startTime, finishTime, iterationDuration, shortestIteration, longestIteration;

    for (int i = 0; i < ITERATION_COUNT; i++) {

        startTime = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        executedTasks = 0;
        remainingTasks = TASK_COUNT;
        additionalTasks = 0;

        executeTasks(tasks);

        int threadResponse;

        for (int procIdx = 0; procIdx < num_of_proc; procIdx++) {

            if (procIdx != procrank) {

                MPI_Send(&procrank, 1, MPI_INT, procIdx, 888, MPI_COMM_WORLD);

                MPI_Recv(&threadResponse, 1, MPI_INT, procIdx, SENDING_TASK_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (threadResponse != NO_TASKS_TO_SHARE) {
                    additionalTasks = threadResponse;
                    memset(tasks, 0, TASK_COUNT);

                    MPI_Recv(tasks, additionalTasks, MPI_INT, procIdx, SENDING_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    pthread_mutex_lock(&mutex);
                    remainingTasks = additionalTasks;
                    pthread_mutex_unlock(&mutex);

                    executeTasks(tasks);
                }
            }

        }
        finishTime = MPI_Wtime();
        iterationDuration = finishTime - startTime;

        MPI_Allreduce(&iterationDuration, &longestIteration, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&iterationDuration, &shortestIteration, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        summaryDisbalance += (longestIteration - shortestIteration)/longestIteration;
    }

    pthread_mutex_lock(&mutex);
    isFinishedExecution = true;
    pthread_mutex_unlock(&mutex);
    int signal = EXECUTOR_FINISHED_WORK;
    MPI_Send(&signal, 1, MPI_INT, procrank, 888, MPI_COMM_WORLD);
    delete[] tasks;
    pthread_exit(nullptr);
}

void* receiver(void* args) {
    int askingProcRank, answer, pendingMessage;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);

    while (!isFinishedExecution) {

        MPI_Recv(&pendingMessage, 1, MPI_INT, MPI_ANY_SOURCE, 888, MPI_COMM_WORLD, &status);

        askingProcRank = pendingMessage;

        if (remainingTasks >= MIN_TASKS_TO_SHARE) {

            pthread_mutex_lock(&mutex);
            answer = remainingTasks / (num_of_proc * 2);
            remainingTasks = remainingTasks / (num_of_proc * 2);
            pthread_mutex_unlock(&mutex);

            MPI_Send(&answer, 1, MPI_INT, askingProcRank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
            MPI_Send(&tasks[TASK_COUNT - answer], answer, MPI_INT, askingProcRank, SENDING_TASKS, MPI_COMM_WORLD);
					//Probably wrong, why last 'answer' tasks again and again?
        }
        else {
            answer = NO_TASKS_TO_SHARE;
            MPI_Send(&answer, 1, MPI_INT, askingProcRank, SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
    }
    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    int threadSupport;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadSupport);
    if (threadSupport != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &procrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc);

    pthread_mutex_init(&mutex, nullptr);
    pthread_attr_t threadAttributes;
    pthread_attr_init(&threadAttributes);

    pthread_attr_setdetachstate(&threadAttributes, PTHREAD_CREATE_JOINABLE);

    double start = MPI_Wtime();

    pthread_create(&threads[0], &threadAttributes, receiver, nullptr);
    pthread_create(&threads[1], &threadAttributes, executor, nullptr);

    pthread_join(threads[0], nullptr);
    pthread_join(threads[1], nullptr);

    pthread_attr_destroy(&threadAttributes);
    pthread_mutex_destroy(&mutex);

    if (procrank == 0) {
        cout << "Proc " << procrank << ": Summary disbalance:" << summaryDisbalance / ITERATION_COUNT * 100 << "%" << endl;
        cout << "Proc " << procrank << ": time taken: " << MPI_Wtime() - start << endl;
    }

    MPI_Finalize();
    return 0;
}
