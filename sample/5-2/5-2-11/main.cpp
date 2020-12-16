#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>

#define NAME_NUM (8)  /* 銘柄数 */
#define DATA_NUM (21) /* 1銘柄あたりの時点数 */

/* 株価データ読み込み */
int stock_array_many[NAME_NUM*DATA_NUM]= {
    #include "stock_array_many.txt"
};

/* 移動平均幅設定 */
#define WINDOW_SIZE (13)

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
    cl_mem memobjOut = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    size_t kernel_code_size;
    char *kernel_src_str;
    float *result;
    cl_int ret;
    FILE *fp;

    int window_num = (int)WINDOW_SIZE;
    int point_num = NAME_NUM * DATA_NUM;
    int data_num = (int)DATA_NUM;
    int name_num = (int)NAME_NUM;

    int i,j;

    /* カーネルソースコード読み込み領域確保 */
    kernel_src_str = (char *)malloc(MAX_SOURCE_SIZE);

    /* 処理結果用メモリ領域確保(ホスト側) */
    result = (float *)malloc(point_num*sizeof(float));

    /* プラットフォーム取得 */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    /* デバイス取得 */
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
                         &ret_num_devices);

    /* コンテキストの作成 */
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    /* コマンドキューの作成 */    
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* カーネルソースコード読み込み(データ並列ベクタ版) */        
    fp = fopen("moving_average_vec4_para.cl", "r");
    kernel_code_size = fread(kernel_src_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* プログラムオブジェクト作成 */
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_str,
                                        (const size_t *)&kernel_code_size, &ret);

    /* カーネルコンパイル */            
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* カーネル作成 */
    kernel = clCreateKernel(program, "moving_average_vec4_para", &ret);

    /* 移動平均入力用バッファ作成(デバイス側) */
    memobjIn  = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               point_num * sizeof(int), NULL, &ret);

    /* 移動平均出力用バッファ作成(デバイス側) */        
    memobjOut = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               point_num * sizeof(float), NULL, &ret);

    /* 移動平均入力用バッファコピー(ホスト→デバイス) */
    ret = clEnqueueWriteBuffer(command_queue, memobjIn, CL_TRUE, 0,
                               point_num * sizeof(int),
                               stock_array_many, 0, NULL, NULL);

    /* カーネル引数設定 */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjIn);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjOut);
    ret = clSetKernelArg(kernel, 2, sizeof(int),    (void *)&data_num);
    ret = clSetKernelArg(kernel, 3, sizeof(int),    (void *)&name_num);
    ret = clSetKernelArg(kernel, 4, sizeof(int),    (void *)&window_num);


    /* データ並列数設定（ワークアイテム数）設定 */
    cl_uint work_dim = 1;
    size_t global_item_size[3];
    size_t local_item_size[3];

    global_item_size[0] = 2; /* 全体のワークアイテム数 = 2 */
    local_item_size[0] = 1;  /* 1ワークグループ当たりのワークアイテム数 = 1 */
    /* --> global_item_size[0] / local_item_size[0] が2になり、ワークグループ数2を指定したことになる */

    /* 移動平均カーネル並列実行 */    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL,
                                 global_item_size, local_item_size,
                                 0, NULL, NULL);

    /* 移動平均出力用バッファコピー(デバイス→ホスト) */
    ret = clEnqueueReadBuffer(command_queue, memobjOut, CL_TRUE, 0,
                              point_num * sizeof(float),
                              result, 0, NULL, NULL);

    /* OpenCLオブジェクト解放 */
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobjIn);
    ret = clReleaseMemObject(memobjOut);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    /* 結果出力 */
    for(i=0; i<data_num; i++) {
        printf( "result[%d]: ", i );
        for(j=0; j<name_num; j++ ) {
            printf( "%f, ", result[i*NAME_NUM+j] );
        }
        printf( "\n" );
    }

    /* ホスト側のメモリ解放 */
    free(result);
    free(kernel_src_str);

    return 0;
}

