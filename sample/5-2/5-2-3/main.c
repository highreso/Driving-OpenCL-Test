#include <stdio.h>
#include <stdlib.h>

/* 株価データ読み込み */
int stock_array1[] = {
    #include "stock_array1.txt"
};

/* 移動平均幅設定 */
#define WINDOW_SIZE (13)

int main(int argc, char *argv[])
{

    float *result;

    int data_num = sizeof(stock_array1) / sizeof(stock_array1[0]);
    int window_num = (int)WINDOW_SIZE;

    int i;
        
    /* 処理結果用メモリ領域確保 */
    result = (float *)malloc(data_num*sizeof(float));

    /* 移動平均呼び出し */
    moving_average(stock_array1,
                   result,
                   data_num,
                   window_num);

    /* 処理結果出力 */
    for(i=0; i<data_num; i++) {
        printf( "result[%d] = %f\n", i, result[i] );
    }

    /* 処理結果用メモリ領域解放 */    
    free(result);

    return 0;
}

