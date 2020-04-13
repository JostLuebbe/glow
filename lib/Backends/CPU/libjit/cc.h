#include "utility.h"

void test2();

int in_bounds(int x, int y, int img_x, int img_y){
    if (x < 0 || y < 0 || x > img_x - 1 || y > img_y - 1) return 0;
    return 1;
}

void convolve(
    int img_x,
    int img_y,
    int** img,
    int ker_x,
    int ker_y,
    int** kernel,
    int stride,
    int** res
)
{
    for (int y = 0; y < img_y; y+=stride){
        for (int x = 0; x < img_x; x+=stride){
            int sum = 0;
            for (int r = -(ker_y / 2); r <= (ker_y / 2); r++){
                for (int c = -(ker_x / 2); c <= (ker_x / 2); c++){
                    if (in_bounds(x + c, y + r, img_x, img_y)) {
                        sum += img[y + r][x + c] * kernel[r + (ker_y / 2)][c + (ker_x / 2)];
                    }
                }
            }
            res[(y / stride)][(x / stride)] = sum;
        }
    }
}

void convolve_offset(
    int img_x,
    int img_y,
    int** img,
    int ker_x,
    int ker_y,
    int** kernel,
    int stride,
    int** res
)
{
    for (int y = 1; y < img_y; y+=stride){
        for (int x = 1; x < img_x; x+=stride){
            int sum = 0;
            for (int r = -(ker_y / 2); r <= (ker_y / 2); r++){
                for (int c = -(ker_x / 2); c <= (ker_x / 2); c++){
                    if (in_bounds(x + c, y + r, img_x, img_y)) {
                        sum += img[y + r][x + c] * kernel[r + (ker_y / 2)][c + (ker_x / 2)];
                    }
                }
            }
            res[(y / stride)][(x / stride)] = sum;
        }
    }

}
