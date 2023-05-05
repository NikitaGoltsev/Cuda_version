#include <iostream>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda.h> 
#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
//#include "sub.cuh" 
#include <chrono>


// поддержка double
#define LF_SUP

#ifdef LF_SUP
#define TYPE double
#define ABS fabs
#define MAX fmax
#define CAST std::stod
#else
#define TYPE floatпш
#define ABS fabsf
#define MAX fmaxf
#define CAST std::stof
#endif

//индексация по фортрану
#define IDX2C(i, j, ld) (((j)*(ld))+(i))


// функция инициализации сетки
void initArr(TYPE *A, int n)
{
    //заполнение углов сетки
    A[IDX2C(0, 0, n)] = 10.0;
    A[IDX2C(0, n - 1, n)] = 20.0;
    A[IDX2C(n - 1, 0, n)] = 20.0;
    A[IDX2C(n - 1, n - 1, n)] = 30.0;


    //заполнение краёв сетки
    for (int i{1}; i < n - 1; ++i)
    {
        A[IDX2C(0,i,n)] = 10 + (i * 10.0 / (n - 1));
        A[IDX2C(i,0,n)] = 10 + (i * 10.0 / (n - 1));
        A[IDX2C(n-1,i,n)] = 20 + (i * 10.0 / (n - 1));
        A[IDX2C(i,n-1,n)] = 20 + (i * 10.0 / (n - 1));
    }
  
}

//функция печати массива на гпу
 void printArr(TYPE *A, int n)
{
    for (int i {0}; i < n; ++i)
    {
        for (int j {0}; j < n; ++j)
        {

            printf("%lf ", A[IDX2C(i,j,n)]);
        }
        std::cout<<std::endl;
    }
    
}


// Шаг алгоритма
__global__ void Step(const double* A, double* Anew, int* dev_n){
    //вычисление ячейки
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    //проверка границ
    if (j == 0 || i == 0 || i == *dev_n-1 || j == *dev_n-1) return;
    //среднее по соседним элементам
    Anew[IDX2C(j, i, *dev_n)] = 0.25 * (A[IDX2C(j, i+1, *dev_n)] + A[IDX2C(j, i-1, *dev_n)] + A[IDX2C(j-1, i, *dev_n)] + A[IDX2C(j+1, i, *dev_n)]);
}


__global__ void reduceBlock(const double *A, const double *Anew, const int n, double *out){
    // создание блока
    typedef cub::BlockReduce<double, 256> BlockReduce; 
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double error = 0;
    // проходим по массивам и находим макс разницу
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
        error = MAX(error, ABS(Anew[i] - A[i]));
    }
    // засовываем максимальную разницу в блок редукции
    double block_max_diff = BlockReduce(temp_storage).Reduce(error, cub::Max());

    // обновление значения
    if (threadIdx.x == 0){
        out[blockIdx.x] = block_max_diff; 
    }
}



//основной цикл программы
void solution(TYPE tol, int iter_max, int n)
{
    //acc_set_device_num(3,acc_device_default);



    //текущая ошибка, счетчик итераций, размер(площадь) сетки
    TYPE error{1.0};
    int iter{0},size{n*n}; 
    

    //alpha - скаляр для вычитания
    //inc - шаг инкремента
    //max_idx - индекс максимального элемента
    TYPE alpha {-1};
    int inc {1}, max_idx { 0};

    //матрицы
    TYPE *A = new TYPE [size], *Anew = new TYPE [size], *Atmp = new TYPE [size];
    
    bool flag {true}; // флаг для обновления значения ошибки на хосте

    //инициализация сеток
    initArr(A, n);
    initArr(Anew, n);

    //указатели на массивы, которые будут лежать на девайсе
    double *dev_A, *dev_Anew, *dev_Atmp;

    //выделение памяти на видеокарте под массивы
    cudaMalloc(&dev_A,size*sizeof(TYPE));
    cudaMalloc(&dev_Anew,size*sizeof(TYPE));
    //cudaMalloc(&dev_Atmp,size*sizeof(TYPE));


    //копирование массивов на видеокарту 
    cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Anew, Anew, size, cudaMemcpyHostToDevice);

    //определение количество потоков на блок
    dim3 threadPerBlock = dim3(32,32); 
    //определение количество блоков на сетку
    dim3 blocksPerGrid = dim3((n + 31) / 32, (n+31)/32);

    //printArr(A,n);
    //std::cout<<"___________________________"<<std::endl;
    
    while (error > tol && iter < iter_max)
    {
        flag = !(iter % n);

        // меняем местами, чтобы не делать swap с доп переменной. работает быстрее
        Step<<<blocksPerGrid, threadPerBlock>>>(A, Anew, n_d); 
        Step<<<blocksPerGrid, threadPerBlock>>>(Anew, A, n_d); 

        if(flag){
            //reduceBlock<<<num_blocks_reduce, THREADS_PER_BLOCK_REDUCE>>>(F, Fnew, size, error_reduction); // по блочно проходим редукцией
            //cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, error_reduction, error, num_blocks_reduce); // проходим редукцией по всем блокам
            ////обновление ошибки на хосте 
            //cudaMemcpy(tmp_err, error, sizeof(double), cudaMemcpyDeviceToHost);
            
        }
        ++iter;   
    }


    std::cout << "Iterations: " << iter << std::endl<< "Error: " << error << std::endl;
    
    cudaFree(dev_A);
    cudaFree(dev_Anew);
    //cudaFree(dev_Atmp);

    delete[] A;
    delete[] Anew;
    delete[] Atmp;
}

int main(int argc, char *argv[])
{
    
    TYPE tol{1e-6};
    int iter_max{1000000}, n{128}; // значения для отладки, по умолчанию инициализировать нулями

    //парсинг командной строки
    std::string tmpStr;
    //-t - точность
    //-n - размер сетки
    //-i - кол-во итераций
    for (int i{1}; i < argc; ++i)
    {
        tmpStr = argv[i];
        if (!tmpStr.compare("-t"))
        {
            tol = CAST(argv[i + 1]);
            ++i;
        }

        if (!tmpStr.compare("-i"))
        {
            iter_max = std::stoi(argv[i + 1]);
            ++i;
        }

        if (!tmpStr.compare("-n"))
        {
            n = std::stoi(argv[i + 1]);
            ++i;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    solution(tol,iter_max,n);
    auto end = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end).count();
    std::cout<<"Time (ms): "<<microseconds/1000<<std::endl;
}