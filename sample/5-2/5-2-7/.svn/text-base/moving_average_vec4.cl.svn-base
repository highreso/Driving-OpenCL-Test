__kernel void moving_average_vec4(__global int4  *values,
                                  __global float4 *average,
                                  int length,
                                  int width)
{
    int i;
    int4 add_value; /* 4銘柄分の加算結果が入る */

    /* width-1 番目の加算処理(4銘柄同時) */
    add_value = (int4)0;
    for( i = 0; i < width; i++ ) {
        add_value += values[i];
    }
    average[width-1] = convert_float4(add_value);

    /* width ～ length-1 番目の加算処理(4銘柄同時) */
    for( i = width; i < length; i++ ) {
        add_value = add_value - values[i-width] + values[i];
        average[i] = convert_float4(add_value);
    }

    /* 0 ～ width -2 番目はクリア(4銘柄同時) */
    for( i = 0; i < width-1; i++ ) {
        average[i] = (float4)(0.0f);
    }

    /* width-1 ～ length-1 番目の平均結果算出(4銘柄同時) */
    for( i = width-1; i < length; i++ ) {
        average[i] /= (float4)width;
    }
}
