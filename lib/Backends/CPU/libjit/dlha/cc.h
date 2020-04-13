#include "utility.h"

void dlha_print2();

int in_bounds(int x, int y, int img_x, int img_y);

void convolve(
    int img_x,
    int img_y,
    int** img,
    int ker_x,
    int ker_y,
    int** kernel,
    int stride,
    int** res
);

void convolve_offset(
    int img_x,
    int img_y,
    int** img,
    int ker_x,
    int ker_y,
    int** kernel,
    int stride,
    int** res
);
