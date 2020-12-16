
#include <cassert>
#include <cstring>
#include <iostream>

#include <CL/cl.h>

#define USING_SUB_DEVICE

int main(int argc, char *argv[])
{
	cl_int ret;

	/* get platform ID */
	cl_uint num_platform;
	cl_platform_id platform_id;
	ret = clGetPlatformIDs(1, &platform_id, &num_platform);
	assert(CL_SUCCESS == ret);

#if defined USING_SUB_DEVICE
	/* get device IDs */
	cl_uint num_device;
	cl_device_id device_id;
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_device);

    assert(CL_SUCCESS == ret);

    /* get number of compute unit */
    cl_uint num_cu;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_cu, NULL);
    assert(CL_SUCCESS == ret);
    
	/* partition devices */
    cl_device_id device_ids[2];
    const cl_device_partition_property props[] = {CL_DEVICE_PARTITION_EQUALLY, num_cu/2, 0};
    ret = clCreateSubDevices(device_id, props, 2, device_ids, NULL);
    assert(CL_SUCCESS == ret);
#else
	/* get device IDs */
	cl_uint num_device;
	cl_device_id device_ids[2];
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 2, device_ids, &num_device);
    assert(CL_SUCCESS == ret);
#endif

	/* create context and command queue */
	cl_context context;
	cl_command_queue command_queues[2];

	context = clCreateContext(NULL, 2, device_ids, NULL, NULL, &ret);
	assert(CL_SUCCESS == ret);

	for (unsigned int i=0; i<2; ++i)
	{
		command_queues[i] = clCreateCommandQueue(context, device_ids[i], 0, &ret);
		assert(CL_SUCCESS == ret);
	}

    /* build kernel */
    const char *source = 
        "#pragma OPENCL EXTENSION cl_amd_printf : enable\n"
        "__kernel\n"
        "void my_kernel(int i)\n"
        "{\n"
        "printf(\"I am device %d.\\n\", i);\n"
        "}\n";

    size_t source_len = strlen(source);
    cl_program program = clCreateProgramWithSource(context, 1, &source, &source_len, &ret);
    assert(CL_SUCCESS == ret);

    ret = clBuildProgram(program, 2, device_ids, NULL, NULL, NULL);
    assert(CL_SUCCESS == ret);

	cl_kernel kernel = clCreateKernel(program, "my_kernel", &ret);
    assert(CL_SUCCESS == ret);

    /* execute */
 	for (unsigned int i=0; i<2; ++i)
	{
        clSetKernelArg(kernel, 0, sizeof(cl_int), &i);
        assert(CL_SUCCESS == ret);

        clEnqueueTask(command_queues[i], kernel, 0, NULL, NULL);
        assert(CL_SUCCESS == ret);
	}   

 	for (unsigned int i=0; i<2; ++i)
	{
		clFinish(command_queues[i]);
        assert(CL_SUCCESS == ret);
	}

    /* finalizing */
    clReleaseKernel(kernel);

 	for (unsigned int i=0; i<2; ++i)
	{
		clReleaseCommandQueue(command_queues[i]);
	}   
    
    clReleaseContext(context);
	
    return 0;
}
