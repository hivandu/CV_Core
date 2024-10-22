'''
Author: Hivan Du
mail: doo@hivan.me
LastEditors: Hivan Du
LastEditTime: 2024-09-21 17:47:52
'''
import pycuda.autoinit
from pycuda.compiler import SourceModule

kernel_code = r'''
__global__ void print_id(void)
{
    //printf('gradIdx.x = %d,blockIdx.x = %d; threadIdx.x = %d;\n', gridIdx.x,blockIdx.x, threadIdx.x);
    printf('blockIdx.x = %d; threadIdx.x = %d; threadIdx.y = %d; threadIdx.z = %d; \n', blockIdx.x, threadIdx.x,threadIdx.y,threadIdx.z);
}
'''
mod = SourceModule(kernel_code)
print_id = mod.get_function('print_id')
# block 内是线程
# 线程组成block,block组成grid
print_id(grid=(2,1,1), block=(6,3,1))
