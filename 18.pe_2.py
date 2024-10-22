'''
Author: Hivan Du
mail: doo@hivan.me
LastEditors: Hivan Du
LastEditTime: 2024-09-21 17:48:11
'''
# 对 cuda 进行初始化
import pycuda.autoinit
# SourceMode C++ 源码编译为 Python
from pycuda.compiler import SourceModuce

kernel_code = r'''
__global__ void hello_from_gpu(void)
{
    printf('[%d, %d] Hello World from the GPU!\n', threadIdx.x, blockIdx.x);
}
__global__ void hello_from_gpu_2(void)
{
    printf('[%d, %d] Hello World from the GPU! the second func\n', threadIdx.x, blockIdx.x);
}
'''

# 编译
mod = SourceModuce(kernel_code)
# 获取函数
hello_from_gpu = mod.get_function('hello_from_gpu_2')
# 利用多线程来调用我们定义的函数
hello_from_gpu(block=(1,1,8))

# 核函数的执行次数就是里面的数字的乘积。那么你可能要有一个疑问并行(3,4,5)为什么不直接写 60？这是由于并行经常被用于处理 2D、3D 问题，这样写参数就很方便
# 并行化体现在哪里?
# https://zhuanlan.zhihu.com/p/125598914