import sys
from time import time
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

import pycuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.scan import InclusiveScanKernel
from pycuda.reduction import ReductionKernel

InteractiveShell.ast_node_interactivity = "all"
print(f'The version of PyCUDA: {pycuda.VERSION}')
print(f'The version of Python: {sys.version}')
import pdb
pdb.set_trace()
def query_device():
    drv.init()
    print('CUDA device query (PyCUDA version) \n')
    print(f'Detected {drv.Device.count()} CUDA Capable device(s) \n')
    for i in range(drv.Device.count()):

        gpu_device = drv.Device(i)
        print(f'Device {i}: {gpu_device.name()}')
        compute_capability = float( '%d.%d' % gpu_device.compute_capability() )
        print(f'\t Compute Capability: {compute_capability}')
        print(f'\t Total Memory: {gpu_device.total_memory()//(1024**2)} megabytes')

        # The following will give us all remaining device attributes as seen
        # in the original deviceQuery.
        # We set up a dictionary as such so that we can easily index
        # the values using a string descriptor.

        device_attributes_tuples = gpu_device.get_attributes().items()
        device_attributes = {}

        for k, v in device_attributes_tuples:
            device_attributes[str(k)] = v

        num_mp = device_attributes['MULTIPROCESSOR_COUNT']

        # Cores per multiprocessor is not reported by the GPU!
        # We must use a lookup table based on compute capability.
        # See the following:
        # http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

        cuda_cores_per_mp = { 5.0 : 128, 5.1 : 128, 5.2 : 128, 6.0 : 64, 6.1 : 128, 6.2 : 128, 7.0 :128}[compute_capability]

        print(f'\t ({num_mp}) Multiprocessors, ({cuda_cores_per_mp}) CUDA Cores / Multiprocessor: {num_mp*cuda_cores_per_mp} CUDA Cores')

        device_attributes.pop('MULTIPROCESSOR_COUNT')

        for k in device_attributes.keys():
            print(f'\t {k}: {device_attributes[k]}')



query_device()

host_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
device_data = gpuarray.to_gpu(host_data)
# 不用写kernel函数
device_data_x2 = 2 * device_data
host_data_x2 = device_data_x2.get()
print(host_data_x2)

# gpuarray 和 get()  使用gpu;
# 部署模型得一个方法
x_host = np.array([1, 2, 3], dtype=np.float32)
y_host = np.array([1, 1, 1], dtype=np.float32)
z_host = np.array([2, 2, 2], dtype=np.float32)
x_device = gpuarray.to_gpu(x_host)
y_device = gpuarray.to_gpu(y_host)
z_device = gpuarray.to_gpu(z_host)

x_host + y_host
(x_device + y_device).get()

x_host ** z_host
(x_device ** z_device).get()

x_host / x_host
(x_device / x_device).get()

z_host - x_host
(z_device - x_device).get()

z_host / 2
(z_device / 2).get()

x_host - 1
(x_device - 1).get()



# 性能比较
def simple_speed_test():
    host_data = np.float32(np.random.random(50000000))

    t1 = time()
    host_data_2x =  host_data * np.float32(2)
    t2 = time()

    print(f'total time to compute on CPU: {t2 - t1}')

    device_data = gpuarray.to_gpu(host_data)

    t1 = time()
    device_data_2x =  device_data * np.float32(2)
    t2 = time()

    from_device = device_data_2x.get()

    print(f'total time to compute on GPU: {t2 - t1}')
    print(f'Is the host computation the same as the GPU computation? : {np.allclose(from_device, host_data_2x)}')

simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
simple_speed_test()
import pdb
pdb.set_trace()



gpu_2x_ker = ElementwiseKernel(
        "float *in, float *out",
        "out[i] = 2 * in[i];",
        "gpu_2x_ker"
    )

def elementwise_kernel_example():
    host_data = np.float32(np.random.random(50000000))
    t1 = time()
    host_data_2x = host_data * np.float32(2)
    t2 = time()
    print(f'total time to compute on CPU: {t2 - t1}')

    device_data = gpuarray.to_gpu(host_data)
    # allocate memory for output
    device_data_2x = gpuarray.empty_like(device_data)

    t1 = time()
    gpu_2x_ker(device_data, device_data_2x)
    t2 = time()
    from_device = device_data_2x.get()
    print(f'total time to compute on GPU: {t2 - t1}')
    print(f'Is the host computation the same as the GPU computation? : {np.allclose(from_device, host_data_2x)}')

elementwise_kernel_example()
elementwise_kernel_example()
elementwise_kernel_example()
elementwise_kernel_example()
elementwise_kernel_example()

import pdb
pdb.set_trace()

seq = np.array([1, 2, 3, 4], dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
sum_gpu = InclusiveScanKernel(np.int32, "a+b")
print(sum_gpu(seq_gpu).get())
print(np.cumsum(seq))
simple_speed_test()
simple_speed_test()
simple_speed_test()

seq = np.array([1,100,-3,-10000, 4, 10000, 66, 14, 21], dtype=np.int32)
seq_gpu = gpuarray.to_gpu(seq)
max_gpu = InclusiveScanKernel(np.int32, "a > b ? a : b")
seq_max_bubble = max_gpu(seq_gpu)
print(seq_max_bubble)
print(seq_max_bubble.get()[-1])
print(np.max(seq))


a_host = np.array([1, 2, 3], dtype=np.float32)
b_host = np.array([4, 5, 6], dtype=np.float32)
print(a_host.dot(b_host))
dot_prod = ReductionKernel(np.float32, neutral="0", reduce_expr="a+b", map_expr="x[i]*y[i]", arguments="float *x, float *y")
a_device = gpuarray.to_gpu(a_host)
b_device = gpuarray.to_gpu(b_host)
print(dot_prod(a_device, b_device).get())
