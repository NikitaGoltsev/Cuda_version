/*
 * Решение уравнения теплопроводности (пятиточечный шаблон) в двумерной области на равномерных
 * сетках (128^2, 256^2, 512^2, 1024^2). Граничные условия – линейная интерполяция между
 * углами области. Значение в углах – 10, 20, 30, 20.
 * 
 * Параметры (точность, размер сетки, количество итераций) задаваются через
 * параметры командной строки.
 * 
 * Вывод программы - количество итераций и достигнутое значение ошибки.
 * 
 * Операция редукции (подсчет максимальной ошибки) реализована с использованием библиотеки CUB.
*/


#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include "cuda_runtime.h"
#include <cub/cub.cuh>

#ifdef _FLOAT
#define T float
#define MAX std::fmaxf
#define STOD std::stof
#else
#define T double
#define MAX std::fmax
#define STOD std::stod
#endif

// Макрос индексации с 0
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// Макрос проверки статуса операции CUDA
#define CUDA_CHECK(err)                                                        \
    {                                                                          \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess)                                               \
        {                                                                      \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    }

// Вывести свойства девайса
void print_device_properties(void)
{
    cudaDeviceProp deviceProp;
    if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, 0))
    {
        printf("Warp size in threads is %d.\n", deviceProp.warpSize);
        printf("Maximum size of each dimension of a block is %d, %d, %d.\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid is %d, %d, %d.\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum resident threads per multiprocessor is %d.\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of resident blocks per multiprocessor is %d.\n", deviceProp.maxBlocksPerMultiProcessor);
        printf("_____________________________________________________________________________________________\n");
    }
}

// Вывести значения двумерного массива на gpu
void print_array_gpu(T *A, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            printf("%.2f\t", A[IDX2C(i, j, size)]);
        }
        printf("\n");
    }
    printf("\n");
}

// Инициализация матрицы, чтобы подготовить ее к основному алгоритму
void initialize_array(T *A, int size)
{
    // Заполнение углов матрицы значениями
    A[IDX2C(0, 0, size)] = 10.0;
    A[IDX2C(0, size - 1, size)] = 20.0;
    A[IDX2C(size - 1, 0, size)] = 20.0;
    A[IDX2C(size - 1, size - 1, size)] = 30.0;

    // Заполнение периметра матрицы
    T step = 10.0 / (size - 1);

    for (int i = 1; i < size - 1; ++i)
    {
        T addend = step * i;
        A[IDX2C(0, i, size)] = A[IDX2C(0, 0, size)] + addend;               // horizontal
        A[IDX2C(size - 1, i, size)] = A[IDX2C(size - 1, 0, size)] + addend; // horizontal
        A[IDX2C(i, 0, size)] = A[IDX2C(0, 0, size)] + addend;               // vertical
        A[IDX2C(i, size - 1, size)] = A[IDX2C(0, size - 1, size)] + addend; // vertical
    }
}

// Посчитать матрицу
__global__ void calculate_matrix(T *Anew, T *A, uint32_t size)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    // Граница или выход за границы массива - ничего не делать
    if (i >= size - 1 || j >= size - 1 || i == 0 || j == 0)
        return;

    Anew[IDX2C(i, j, size)] = (A[IDX2C(i + 1, j, size)] + A[IDX2C(i - 1, j, size)] + A[IDX2C(i, j - 1, size)] + A[IDX2C(i, j + 1, size)]) * 0.25;
}

// O = |A-B|
__global__ void count_matrix_difference(T *matrixA, T *matrixB, T *outputMatrix, uint32_t size)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t j = blockDim.y * blockIdx.y + threadIdx.y;

    // Выход за границы массива или периметр - ничего не делать
    if (i >= size - 1 || j >= size - 1 || i == 0 || j == 0)
        return;

    uint32_t idx = IDX2C(i, j, size);
    outputMatrix[idx] = std::abs(matrixB[idx] - matrixA[idx]);
}

void calculate(const int net_size = 128, const int iter_max = 1e6, const T accuracy = 1e-6, const bool res = false)
{
    CUDA_CHECK(cudaSetDevice(1));
    const size_t vec_size = net_size * net_size;

    uint32_t threads_in_block = MIN(net_size, 32);                      // Потоков в одном блоке (32 * 32 максимум)
    uint32_t block_in_grid = ceil((double)net_size / threads_in_block); // Блоков в сетке (size / 32 максимум)

    dim3 blockPerGrid = dim3(block_in_grid,block_in_grid), threadPerBlock = dim3(threads_in_block,threads_in_block);

    // Матрица на хосте (нужна только для инициализации и вывода) [Pinned]
    T *A;

    CUDA_CHECK(cudaMallocHost(&A, sizeof(T) * vec_size));

    // Инициализация матрицы
    initialize_array(A, net_size);

    // Создание 2-х матриц на девайсе
    T *A_dev, *Anew_dev;
    CUDA_CHECK(cudaMalloc(&A_dev, sizeof(T) * vec_size));
    CUDA_CHECK(cudaMalloc(&Anew_dev, sizeof(T) * vec_size));

    // Поток для копирования
    cudaStream_t memory_stream;
    CUDA_CHECK(cudaStreamCreate(&memory_stream));

    // Скопировать матрицу с хоста на матрицы на девайсе
    CUDA_CHECK(cudaMemcpy(A_dev, A, sizeof(T) * vec_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Anew_dev, A, sizeof(T) * vec_size, cudaMemcpyHostToDevice));

    // Текущая ошибка
    T *error, *error_dev;
    CUDA_CHECK(cudaMallocHost(&error, sizeof(T)));
    CUDA_CHECK(cudaMalloc(&error_dev, sizeof(T)));
    *error = accuracy + 1;

    // Матрица ошибок
    T *A_err;
    CUDA_CHECK(cudaMalloc(&A_err, sizeof(T) * vec_size));

    // Временный буфер для редукции и его размер
    T *reduction_bufer = NULL;
    uint64_t reduction_bufer_size = 0;

    // Первый вызов, чтобы предоставить количество байтов, необходимое для временного хранения, необходимого CUB.
    cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size);

    // Выделение памяти под буфер
    CUDA_CHECK(cudaMalloc(&reduction_bufer, reduction_bufer_size));

    // Граф
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Сокращение количества обращений к CPU. Больше сетка - реже стоит сверять значения.
    uint32_t num_skipped_checks = (iter_max < net_size) ? iter_max : net_size;
    num_skipped_checks += num_skipped_checks % 2; // Привести к четному числу

    for (uint32_t k = 0; k < num_skipped_checks; k += 2)
    {
        calculate_matrix<<<blockPerGrid, threadPerBlock, 0, stream>>>(A_dev, Anew_dev, net_size);
        calculate_matrix<<<blockPerGrid, threadPerBlock, 0, stream>>>(Anew_dev, A_dev, net_size);
        // swap не работает
    }

    count_matrix_difference<<<blockPerGrid, threadPerBlock, 0, stream>>>(A_dev, Anew_dev, A_err, net_size);
   
    // Найти максимум и положить в error_dev - аналог reduction (max : error_dev) в OpenACC
    cub::DeviceReduce::Max(reduction_bufer, reduction_bufer_size, A_err, error_dev, vec_size, stream);

    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    // Счетчик итераций
    int iter = 0;

    // Вывод
    // if (res)
    //    print_array_gpu(A, net_size);

    for (iter = 0; iter < iter_max && *error > accuracy; iter += num_skipped_checks)
    {
        // Запуск графа
        CUDA_CHECK(cudaGraphLaunch(instance, stream));

        // Синхронизация потока
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Копировать ошибку с девайса на хост
        CUDA_CHECK(cudaMemcpy(error, error_dev, sizeof(T), cudaMemcpyDeviceToHost));
    }

    std::cout << "Iter: " << iter << " Error: " << *error << std::endl;

    // Вывод
    if (res)
    {
        CUDA_CHECK(cudaMemcpyAsync(A, A_dev, sizeof(T) * vec_size, cudaMemcpyDeviceToHost, memory_stream));
        print_array_gpu(A, net_size);
    }

    // Освобождение памяти
    CUDA_CHECK(cudaFree(reduction_bufer));
    CUDA_CHECK(cudaFree(A_err));
    CUDA_CHECK(cudaFree(A_dev));
    CUDA_CHECK(cudaFree(Anew_dev));
    CUDA_CHECK(cudaFreeHost(A));
    CUDA_CHECK(cudaFreeHost(error));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(memory_stream));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

int main(int argc, char *argv[])
{
    // Начать отсчет времени работы
    auto begin_main = std::chrono::steady_clock::now();

    // Парсинг аргументов командной строки
    int net_size = 128, iter_max = (int)1e6;
    T accuracy = 1e-6;
    bool res = false;
    for (int arg = 1; arg < argc; arg++)
    {
        std::string str = argv[arg];
        if (!str.compare("-res"))
            res = true;
        else
        {
            arg++;
            if (!str.compare("-a"))
                accuracy = STOD(argv[arg]);
            else if (!str.compare("-i"))
                iter_max = (int)std::stod(argv[arg]);
            else if (!str.compare("-s"))
                net_size = std::stoi(argv[arg]);
            else
            {
                std::cout << "Wrong args!\n";
                return -1;
            }
        }
    }

    // Заупстить решение задачи
    calculate(net_size, iter_max, accuracy, res);

    // Посчитать время выполнения
    auto end_main = std::chrono::steady_clock::now();
    int time_spent_main = std::chrono::duration_cast<std::chrono::milliseconds>(end_main - begin_main).count();
    std::cout << "The elapsed time is:\nmain\t\t\t" << time_spent_main << " ms\n";
    return 0;
}