void moving_average_float(float *values,
                          float *average,
                          int length,
                          int width)
{
    int i,j;
    float add_value;

    /* 0 ～ width -2 番目はクリア */
    for( i = 0; i < width-1; i++ ) {
        average[i] = 0.0f;
    }

    /* width-1 ～ length-1 番目の移動平均 */
    for( i = width-1; i < length; i++ ) {
        add_value = 0.0f;
        for( j = 0; j < width; j++ ) {
            add_value += values[i - j];
        }
        average[i] = add_value / (float)width;
    }
}

