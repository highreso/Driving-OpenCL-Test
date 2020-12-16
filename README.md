# amd1dlでOpenACCでGPU駆動させる
## 概要
amd1dlインスタンスでOpenACCで実際にGPUを駆動させることを目指す
ROCm + OpenCLをOpenACCから動作させるのが最終目標
`user@jupyterlab-test`インスタンスで検証中

## 動いたC++コード
https://rocmdocs.amd.com/en/latest/Programming_Guides/Opencl-programming-guide.html#example

export ROCMOPENCL=/opt/rocm/opencl/

```cpp
//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#define NWITEMS 512
// A simple memset kernel
const char *source =
"kernel void memset(   global uint *dst )             \n"
"{                                                    \n"
"    dst[get_global_id(0)] = get_global_id(0);        \n"
"}                                                    \n";

int main(int argc, char ** argv)
{
  // 1. Get a platform.
  cl_platform_id platform;
  clGetPlatformIDs( 1, &platform, NULL );

  // 2. Find a gpu device.
  cl_device_id device;
  clGetDeviceIDs( platform,
                  CL_DEVICE_TYPE_GPU,
                  1,
                  &device, NULL);

  // 3. Create a context and command queue on that device.
  cl_context context = clCreateContext( NULL,
                                        1,
                                        &device,
                                        NULL, NULL, NULL);

  cl_command_queue queue = clCreateCommandQueue( context,
                                                 device,
                                                 0, NULL );

  // 4. Perform runtime source compilation, and obtain kernel entry point.
  cl_program program = clCreateProgramWithSource( context,
                                                  1,
                                                  &source,
                                                  NULL, NULL );

  clBuildProgram( program, 1, &device, NULL, NULL, NULL );

  cl_kernel kernel = clCreateKernel( program, "memset", NULL );

  // 5. Create a data buffer.
  cl_mem buffer = clCreateBuffer( context,
                                  CL_MEM_WRITE_ONLY,
                                  NWITEMS * sizeof(cl_uint),
                                  NULL, NULL );

  // 6. Launch the kernel. Let OpenCL pick the local work size.
  size_t global_work_size = NWITEMS;
  clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);

  clEnqueueNDRangeKernel( queue,
                          kernel,
                          1,
                          NULL,
                          &global_work_size,
                          NULL,
                          0,
                          NULL, NULL);

  clFinish( queue );

  // 7. Look at the results via synchronous buffer map.
  cl_uint *ptr;
  ptr = (cl_uint *) clEnqueueMapBuffer( queue,
                                        buffer,
                                        CL_TRUE,
                                        CL_MAP_READ,
                                        0,
                                        NWITEMS * sizeof(cl_uint),
                                        0, NULL, NULL, NULL );

  int i;

  for(i=0; i < NWITEMS; i++)
      printf("%d %d\n", i, ptr[i]);

  return 0;
}
```
を`Template.cpp`として保存

g++ -o Template.o -c Template.cpp -I$ROCMOPENCL/include

g++ -o Template Template.o -lOpenCL -L$ROCMOPENCL/lib/x86_64

./Template


## PyOpenCL入れてみた
pip install pyopencl
export PYOPENCL_COMPILER_OUTPUT=1



##
dpkg -l | grep opencl
ldconfig -p