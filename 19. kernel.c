__global__ void convolutionGPU(float *output, float *input, float *filter, int filter_radius, int input_width, int input_height) {
    const int filter_width = 2 * filter_radius + 1;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int idx = row + input_width * col;

    if (row < input_width && col < input_height) {  // 边界检查
        float sum = 0;
        float value = 0;
        for (int i = -filter_radius; i <= filter_radius; i++) {
            for (int j = -filter_radius; j <= filter_radius; j++) {
                if ((col + j) < 0 || (row + i) < 0 || (row + i) > (input_width - 1) || (col + j) > (input_height - 1)) {
                    value = 0;
                } else {
                    value = input[idx + i + j * input_width];
                    sum += value * filter[(i + filter_radius) + (j + filter_radius) * filter_width];
                }
            }
        }
        output[idx] = sum;
    }
}