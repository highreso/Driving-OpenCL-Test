#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <malloc.h>

#include <CL/cl.h>

#ifdef _MSC_VER
#define ALIGNED_MALLOC(size, alignment) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr)               _aligned_free(ptr)
#elif __GNUC__
#define ALIGNED_MALLOC(size, alignment) memalign(alignment, size)
#define ALIGNED_FREE(ptr)               free(ptr)
#else
#error "unsupported compiler"
#endif

#define IMAGE_W (8)
#define IMAGE_H (8)

int main(int argc, char *argv[])
{
    cl_int ret;
    
    /* get platform ID */
    cl_platform_id platform_id;
    ret = clGetPlatformIDs(1, &platform_id, NULL);
    assert(CL_SUCCESS == ret);

    /* get device IDs */
    cl_device_id device_id;
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
	assert(CL_SUCCESS == ret);
    
    /* create context */
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    assert(CL_SUCCESS == ret);

    /* create command queue */
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    assert(CL_SUCCESS == ret);

    /* create image object */
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNSIGNED_INT8;
    
    cl_image_desc desc;
	memset(&desc, 0, sizeof(desc));
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width  = IMAGE_W;
    desc.image_height = IMAGE_H;
    
    cl_mem image = clCreateImage(context, 0, &format, &desc, NULL, &ret);
    assert(CL_SUCCESS == ret);

	/* filling background image */
    {
        const size_t origin[] = {0, 0, 0};
        const size_t region[] = {IMAGE_W, IMAGE_H, 1};
 		cl_uchar4 fill_color;
		fill_color.s[0] = 0;
		fill_color.s[1] = 0;
		fill_color.s[2] = 0;
		fill_color.s[3] = 0;
        ret = clEnqueueFillImage(command_queue, image, &fill_color, origin, region, 0, NULL, NULL);
        assert(CL_SUCCESS == ret);
    }

    /* filling front image */
    {
        const size_t origin[] = {(IMAGE_W*1)/4, (IMAGE_H*1)/4, 0};
        const size_t region[] = {(IMAGE_W*2)/4, (IMAGE_H*2)/4, 1};
        cl_uchar4 fill_color;
		fill_color.s[0] = 255;
		fill_color.s[1] = 0;
		fill_color.s[2] = 0;
		fill_color.s[3] = 0;
        ret = clEnqueueFillImage(command_queue, image, &fill_color, origin, region, 0, NULL, NULL);
        assert(CL_SUCCESS == ret);
    }

    /* reading image */
    cl_uchar *data = NULL;
    {
        size_t num_channels = 1;
        data = static_cast<cl_uchar*>(ALIGNED_MALLOC(IMAGE_W*IMAGE_H*sizeof(cl_uchar), num_channels*sizeof(cl_uchar)));
		assert(NULL != data);
		std::fill(&data[0], &data[IMAGE_W*IMAGE_H], 128);
        
        const size_t origin[] = {0, 0, 0};
        const size_t region[] = {IMAGE_W, IMAGE_H, 1};
        ret = clEnqueueReadImage(command_queue, image, CL_TRUE, origin, region, IMAGE_W*sizeof(cl_uchar), 0, data, 0, NULL, NULL);
        assert(CL_SUCCESS == ret);
    }

    /* print image */
    for (unsigned int h=0; h<IMAGE_H; ++h)
    {
        for (unsigned int w=0; w<IMAGE_W; ++w)
        {
            std::cout << std::setw(5) << std::right << static_cast<int>(data[h*IMAGE_W+w]);
        }
        std::cout << std::endl;
    }

    /* finalizing */
    ALIGNED_FREE(data);

    clReleaseMemObject(image);
    
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
