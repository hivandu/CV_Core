'''
Author: Hivan Du
mail: doo@hivan.me
LastEditors: Hivan Du
LastEditTime: 2024-09-22 00:25:48
'''
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy

mod = SourceModule(r'''
void __global__ add(const float *x, const float *y, float *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n]  = x[n] + y[n];
}
void __global__ mul(const float *x, const float *y, float *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] * y[n];
}
                   
/*  gpuMatMultKernel: GPU 下矩阵乘法核函数
*   a: 第一个矩阵指针，表示 a[M][N]
*   b: 第二个矩阵指针，表示 b[N][S]
*   result: 结果矩阵，表示 result[M][S]
*/
//__global__ void gpuMatMultKernel(const int *a, const int *b, int *result, const int M, const int N, const int S)
__global__ void gpuMatMultKernel(float *a, float *b, float *result, int *m, int *n, int *s)
{
    //                  0             10
    // int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    // int threadId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int threadId = threadIdx.x;
    // int threadId = threadIdx.x + threadIdx.y * 5;

    printf('blockIdx.x = %d \n', threadId);
    // M = 3;
    // N = 4;
    // S = 5;
    int M = m[0];
    int N = n[0];
    int S = s[0];

    printf('MNS=%d,%d,%d \n',M,N,S);
    if (threadId < M * S)
    {
        int row = threadId / S;
        // int column = threadId % S;
        int column = threadId - row * S;

        result[threadId] = 0;
        for (int i = 0; i < N; i++)
        {
            result[threadId] += a[row * N + i] * b[i * S + column];
        }
        // printf('blockIdx.x = %d, result = %f \n', row, column);
        printf('row = %d, column = %d, S = %d\n', row, column, S);
    }
}
''')

add = mod.get_function('add')
mul = mod.get_function('mul')
gmm = mod.get_function('gpuMatMultKernel')
num = 6
# 一个 block 计算不完
A = numpy.random.rand(num)
B = numpy.random.rand(num)
C = numpy.zeros(num)
A_GPU = gpuarray.to_gpu(A.astype(numpy.float32))
B_GPU = gpuarray.to_gpu(B.astype(numpy.float32))
C_GPU = gpuarray.to_gpu(B.astype(numpy.float32))

#
add(A_GPU, B_GPU, C_GPU, grid=(2,), block=(4,1,1))
C = C_GPU.get()
print('A=', A)
print('B=', B)
print('C=', C)

mul(A_GPU, B_GPU, C_GPU, grid=(2,), block=(4,1,1))
C = C_GPU.get()
print('A=', A)
print('B=', B)
print('C=', C)

A = numpy.random.rand(30,45)
B = numpy.random.rand(45,50)
C = numpy.random.rand(30,50)
# C = numpy.zeros([10,10])
A_GPU = gpuarray.to_gpu(A.astype(numpy.float32))
B_GPU = gpuarray.to_gpu(B.astype(numpy.float32))
C_GPU = gpuarray.to_gpu(C.astype(numpy.float32))
M = numpy.array([30])
M_GPU = gpuarray.to_gpu(M.astype(numpy.int))
N = numpy.array([45])
N_GPU = gpuarray.to_gpu(N.astype(numpy.int))
S = numpy.array([50])
S_GPU = gpuarray.to_gpu(S.astype(numpy.int))
gmm(A_GPU, B_GPU, C_GPU, M_GPU, N_GPU, S_GPU, grid=(1,), block=(1500,1,1))
# gmm(A_GPU, B_GPU, C_GPU, M_GPU, N_GPU, S_GPU, grid=(1,), block=(3,5,1))

C = C_GPU.get()
print('GMM')
print('A=', A)
print('B=', B)
print('C=', C)

# 线程索引：写好核函数
