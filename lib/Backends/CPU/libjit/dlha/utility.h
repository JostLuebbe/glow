//
// Created by Jost Luebbe on 2020-02-25.
//

#ifndef CONV_TEST_UTILITY_H
#define CONV_TEST_UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define IMG_SIZE 10
#define SW_IMG_SIZE 33
#define SW_RES_SIZE 32
#define RES_SIZE IMG_SIZE-2
void print_matrix(size_t rows, size_t cols, int** a);

int** read_matrix(size_t r, size_t c, FILE *matrix_file);

void hw_fill_img(int hw_img[16][IMG_SIZE][IMG_SIZE],int** sw_img);

void print_matrix_hw_res(int rows, int cols, int result[16][RES_SIZE][RES_SIZE]);

void print_matrix_hw_img(int rows, int cols, int result[16][IMG_SIZE][IMG_SIZE]);

void check_first_conv_result(int rows, int cols, int** sw_res, int hw_res[16][RES_SIZE][RES_SIZE]);

#endif //CONV_TEST_UTILITY_H
