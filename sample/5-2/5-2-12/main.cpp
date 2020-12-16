#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>

#define NAME_NUM (4)   /* 銘柄数 */
#define DATA_NUM (100) /* 1銘柄あたりの時点数 */

/* 株価データ読み込み */
int stock_array_4[NAME_NUM*DATA_NUM]= {
    #include "stock_array_4.txt"
};

/* 移動平均幅設定 */
#define WINDOW_SIZE_13 (13)
#define WINDOW_SIZE_26 (26)


#define MAX_SOURCE_SIZE (0x100000)


int main(void)
{
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobjIn = NULL;
    cl_mem memobjOut13 = NULL;
    cl_mem memobjOut26 = NULL;
    cl_program program = NULL;
    cl_kernel kernel13 = NULL;
    cl_kernel kernel26 = NULL;
    cl_event event13, event26;
    size_t kernel_code_size;
    char *kernel_src_str;
    float *result13;
    float *result26;
    cl_int ret;
    FILE *fp;

    int window_num_13 = (int)WINDOW_SIZE_13;
    int window_num_26 = (int)WINDOW_SIZE_26;
    int point_num = NAME_NUM * DATA_NUM;
    int data_num = (int)DATA_NUM;
    int name_num = (int)NAME_NUM;

    int i,j;

    /* カーネルソースコード読み込み領域確保 */
    kernel_src_str = (char *)malloc(MAX_SOURCE_SIZE);

    /* 処理結果用メモリ領域確保(ホスト側) */
    result13 = (float *)malloc(point_num*sizeof(float)); /* 13週平均分 */
    result26 = (float *)malloc(point_num*sizeof(float)); /* 26週平均分 */

    /* プラットフォーム取得 */    
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    /* デバイス取得 */    
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
                         &ret_num_devices);

    /* コンテキストの作成 */    
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    /* コマンドキューの作成 */        
    command_queue = clCreateCommandQueue(context, device_id,
                                         CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);

    /* カーネルソースコード読み込み(ベクタ版) */
    fp = fopen("moving_average_vec4.cl", "r");
    kernel_code_size = fread(kernel_src_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* プログラムオブジェクト作成 */
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_str,
                                        (const size_t *)&kernel_code_size, &ret);

    /* カーネルコンパイル */    
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* カーネル作成 */
    kernel13 = clCreateKernel(program, "moving_average_vec4", &ret); /* 13週用 */
    kernel26 = clCreateKernel(program, "moving_average_vec4", &ret); /* 26週用 */

    /* 移動平均入力用バッファ作成(デバイス側) */
    memobjIn  = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               point_num * sizeof(int), NULL, &ret);

    /* 移動平均出力用バッファ作成(デバイス側) */    
    memobjOut13 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 point_num * sizeof(float), NULL, &ret); /* 13週用 */
    memobjOut26 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 point_num * sizeof(float), NULL, &ret); /* 26週用 */

    /* 移動平均入力用バッファコピー(ホスト→デバイス) */
    ret = clEnqueueWriteBuffer(command_queue, memobjIn, CL_TRUE, 0,
                               point_num * sizeof(int),
                               stock_array_4, 0, NULL, NULL);

    /* カーネル引数設定 (13週)*/
    ret = clSetKernelArg(kernel13, 0, sizeof(cl_mem), (void *)&memobjIn);
    ret = clSetKernelArg(kernel13, 1, sizeof(cl_mem), (void *)&memobjOut13);
    ret = clSetKernelArg(kernel13, 2, sizeof(int),    (void *)&data_num);
    ret = clSetKernelArg(kernel13, 3, sizeof(int),    (void *)&window_num_13);

    /* 13週移動平均カーネル タスク実行 */    
    ret = clEnqueueTask(command_queue, kernel13, 0, NULL, &event13);

    /* カーネル引数設定 (26週)*/
    ret = clSetKernelArg(kernel26, 0, sizeof(cl_mem), (void *)&memobjIn);
    ret = clSetKernelArg(kernel26, 1, sizeof(cl_mem), (void *)&memobjOut26);
    ret = clSetKernelArg(kernel26, 2, sizeof(int),    (void *)&data_num);
    ret = clSetKernelArg(kernel26, 3, sizeof(int),    (void *)&window_num_26);

    /* 26週移動平均カーネル タスク実行 */    
    ret = clEnqueueTask(command_queue, kernel26, 0, NULL, &event26);

    /* 13週移動平均出力用バッファコピー(デバイス→ホスト) */
    ret = clEnqueueReadBuffer(command_queue, memobjOut13, CL_TRUE, 0,
                              point_num * sizeof(float),
                              result13, 1, &event13, NULL);

    /* 26週移動平均出力用バッファコピー(デバイス→ホスト) */
    ret = clEnqueueReadBuffer(command_queue, memobjOut26, CL_TRUE, 0,
                              point_num * sizeof(float),
                              result26, 1, &event26, NULL);

    /* OpenCLオブジェクト解放 */
    ret = clReleaseKernel(kernel13);
    ret = clReleaseKernel(kernel26);    
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobjIn);
    ret = clReleaseMemObject(memobjOut13);
    ret = clReleaseMemObject(memobjOut26);    
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    /* 結果出力 */
    for(i=window_num_26-1; i<data_num; i++) {
        printf( "result[%d]: ", i );
        for(j=0; j<name_num; j++ ) {
            /* 判定結果 */
            printf( "[%d] ", (result13[i*NAME_NUM+j] >  result26[i*NAME_NUM+j]) );
        }
        printf( "\n" );
    }

    /* ホスト側のメモリ解放 */    
    free(result13);
    free(result26);
    free(kernel_src_str);

    return 0;
}

