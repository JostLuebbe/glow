//
// Created by Jost Luebbe on 2020-02-25.
//

#include "utility.h"

void print_matrix(size_t rows, size_t cols, int** a) {
    printf("[");
    for (size_t r = 0; r<rows; r++){
        if (r == 0)
            printf("[");
        else
            printf(" [");
        for (size_t c = 0; c<cols; c++){
            if (c < cols - 1)
                printf("%d ",  a[r][c]);
            else
                printf("%d",  a[r][c]);
        }
        printf("]");
        if (r < rows - 1)
            printf("\n");
    }
    printf("]");
    printf("\n");
}

int** read_matrix(size_t rows, size_t cols, FILE *matrix_file){

    int** matrix = malloc(sizeof(int*) * rows);

    for (int r = 0; r<rows; r++){
        matrix[r] = malloc(sizeof(int) * cols);
        for (int c = 0; c<cols; c++){
            fscanf(matrix_file, "%d", &matrix[r][c]);
        }
    }

    return matrix;
}




void hw_fill_img(int image[16][IMG_SIZE][IMG_SIZE], int** sw_image){ //good ol matrix padding
    int img_x, img_y, index_x, index_y;
    int fill = 0;
    int i =0;
    for(index_y = 0; index_y < 4;index_y++){
        for(index_x = 0; index_x < 4; index_x++){
            for(img_y = 0; img_y <IMG_SIZE;img_y++){
                for(img_x = 0; img_x < IMG_SIZE; img_x++){
                    if((img_y == 0 && index_y == 0) || (img_x == 0 && index_x == 0) || (img_y == 9 && index_y == 3) || (img_x == 9 && index_x == 3)) //checking where you need to zero pad
                        image[index_x + 4*index_y][img_y][img_x] = 0;
                    else
                        image[index_x + 4*index_y][img_y][img_x] = sw_image[img_y + (index_y * 8) -1 ][(index_x*8) + img_x -1];

                }
            }
        }
    }
}




void print_matrix_hw_res(int rows, int cols, int result[16][RES_SIZE][RES_SIZE]) {
	printf("********************************PRINTING RESULT********************************\n");
	int img_x, img_y, index_x, index_y;
		for(index_y = 0; index_y <rows/2;index_y++){
			for(img_y = 0; img_y <rows;img_y++){
				for(index_x = 0; index_x < cols/2; index_x++){
					for(img_x = 0; img_x < cols; img_x++){
							printf("%d ", result[index_x + 4*index_y][img_y][img_x]);
					}
				}
				printf("\n");
			}
			printf("\n");
		}
		printf("********************************DONE PRINTING RESULT********************************\n");
}

void print_matrix_hw_img(int rows, int cols, int result[16][IMG_SIZE][IMG_SIZE]) {
	printf("********************************PRINTING IMAGE********************************\n");
	int img_x, img_y, index_x, index_y;
	for(index_y = 0; index_y < 4;index_y++){
		for(img_y = 0; img_y <IMG_SIZE;img_y++){
			for(index_x = 0; index_x < 4; index_x++){
				for(img_x = 0; img_x < IMG_SIZE; img_x++){
						printf("%d ", result[index_x + 4*index_y][img_y][img_x]);
				}
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("********************************DONE PRINTING IMAGE********************************\n");
}

void check_first_conv_result(int rows, int cols, int** sw_res, int hw_res[16][RES_SIZE][RES_SIZE]){
	printf("********************************CHECKING RESULTS********************************\n");
	int img_x, img_y, index_x, index_y;
	int err_cnt = 0;
	/*
	for(int i = 0; i<16; i++){
	  for(int j = 0; j< 16; j++){
	  	diff_matrix[i][j] = 0;
	  }
	}
	*/
	 int** diff_matrix = malloc(sizeof(int*) * 32);	//change 32 to RESSIZE
	
	for (int r = 0; r < 32; r++){
        	diff_matrix[r] = malloc(sizeof(int) * 32);
        	for (int c = 0; c < 32; c++){
			diff_matrix[r][c] = 0;
       		}
    	}
	for(index_y = 0; index_y < rows/2;index_y++){
		for(img_y = 0; img_y <rows;img_y++){
			for(index_x = 0; index_x < cols/2; index_x++){
				for(img_x = 0; img_x < cols; img_x++){
					if(hw_res[index_x + 4*index_y][img_y][img_x] != sw_res[img_y + index_y*8][img_x + index_x*8]){ //change 8 to what index_y goes up to, this is only for the sw_res
						printf("index_y: %d ", index_y);
						printf("img_y %d ", img_y);
						printf("index_x: %d ", index_x);
						printf("img_x: %d ", img_x);
						diff_matrix[img_y + index_y*8][img_x + index_x*8] += 1;
						err_cnt += 1;
						printf("error count: %d\n ", err_cnt);
					}
				}
			}
		}
	}

	printf("diff check done\n");
	if(err_cnt > 0){
		printf("err_cnt: %d\n", err_cnt);
		printf("********************************PRINTING DIFF MATRIX********************************\n");
		print_matrix(32, 32, diff_matrix); //if you change the malloc dimensions, change this too
		printf("********************************DONE PRINTING DIFF MATRIX********************************\n");
		}
	else{
		printf("Congrats! No errors!\n");
	}
	printf("********************************DONE CHECKING RESULTS********************************\n");
}
