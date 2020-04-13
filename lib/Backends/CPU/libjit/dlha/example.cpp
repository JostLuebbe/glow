#include "cc.h"
#include "utility.h"
#include <assert.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define READ_CMD  (0x0 << 31)
#define WRITE_CMD (0x1 << 31)

volatile int det_int = 0;

// signal handler for receiving events from hardware driver
void sighandler(int signo) {
    if (signo == SIGIO) {
        det_int++;
        printf("\nInterrupt detected\n");
    }
}


void test_conv(int** img, int** kernel, int** result){
    unsigned long volatile trig, gie, iie, stride;;
    struct sigaction action;
    int fd;

    // install signal handler
    sigemptyset(&action.sa_mask);
    sigaddset(&action.sa_mask, SIGIO);

    action.sa_handler = sighandler;
    action.sa_flags = 0;

    sigaction(SIGIO, &action, NULL);

    // open hardware device (driver)
    fd = open("/dev/fpga", O_RDWR);
    if (fd < 0) {

        printf("Unable to open /dev/fpga.  Ensure it exists!\n");
        return;
    }
    fcntl(fd, F_SETOWN, getpid());
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_ASYNC);

    // enable FPGA interrupts (global and IP)
    ioctl(fd, READ_CMD + 0x1, &gie);
    gie = gie | 0x00000001;
    ioctl(fd, WRITE_CMD + 0x1, &gie);

    iie = 0x1;
    ioctl(fd, WRITE_CMD + 0x2, &iie);

    // writing img and kernel matrices
    int offset = 0x80; //images

    for (int r = 0; r < 9; r++) {
        for (int c = 0; c < 9; c++) {
            ioctl(fd, WRITE_CMD + offset++, &img[r][c]);
        }
    }

    offset = 0x100;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            ioctl(fd, WRITE_CMD + offset++, &kernel[r][c]);
        }
    }

    offset = 0x110;
    stride = 0x02;
    ioctl(fd, WRITE_CMD + offset, &stride);

    // trigger MAC operation
    trig = 0x1;
    ioctl(fd, WRITE_CMD, &trig);

    offset = 0x120;
    // wait for interrupt
    while (!det_int) continue;

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            ioctl(fd, READ_CMD + offset++, &result[r][c]);
        }
    }

    //In the end, close the device driver
    close(fd);
}



void nick_test_conv(int img[16][10][10], int** kernel, int result[16][8][8]){
    unsigned long volatile trig, gie, iie, stride;;
    struct sigaction action;
    int fd;

    // install signal handler
    sigemptyset(&action.sa_mask);
    sigaddset(&action.sa_mask, SIGIO);

    action.sa_handler = sighandler;
    action.sa_flags = 0;

    sigaction(SIGIO, &action, NULL);

    // open hardware device (driver)
    fd = open("/dev/fpga", O_RDWR);
    if (fd < 0) {

        printf("Unable to open /dev/fpga.  Ensure it exists!\n");
        return;
    }
    fcntl(fd, F_SETOWN, getpid());
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_ASYNC);

    // enable FPGA interrupts (global and IP)
    ioctl(fd, READ_CMD + 0x1, &gie);
    gie = gie | 0x00000001;
    ioctl(fd, WRITE_CMD + 0x1, &gie);

    iie = 0x1;
    ioctl(fd, WRITE_CMD + 0x2, &iie);

    // writing img and kernel matrices
    int offset = 0x800; //images

    for(int index = 0; index <16; index++){
    for (int r = 0; r < 10; r++) {
        for (int c = 0; c < 10; c++) {
            ioctl(fd, WRITE_CMD + offset++, &img[index][r][c]);
        }
    }
    }
    offset = 0x1000;
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            ioctl(fd, WRITE_CMD + offset++, &kernel[r][c]);
        }
    }

    offset = 0x1010;
    stride = 0x01;
    ioctl(fd, WRITE_CMD + offset, &stride);

    // trigger MAC operation
    trig = 0x1;
    ioctl(fd, WRITE_CMD, &trig);

    offset = 0x1400;
    // wait for interrupt
    while (!det_int) continue;
    for(int index = 0; index <16; index ++){
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                ioctl(fd, READ_CMD + offset++, &result[index][r][c]);
            }
        }
    }
    //In the end, close the device driver
    close(fd);
}

void dlha_print(){
    printf("IN TEST\n");
    dlha_print2();
}


void run(int** img, int** kernel, size_t img_r, size_t img_c, size_t kernel_r, size_t kernel_c){
    int input_stride = 1;

    int hw_img[16][10][10];
    int hw_res[16][8][8];
    hw_fill_img(hw_img, img);

    print_matrix(img_r, img_c, img);

    print_matrix(kernel_r, kernel_c, kernel);

    int** result = malloc(sizeof(int*) * img_r);
    for (int r = 0; r<img_r; r++){
        result[r] = malloc(sizeof(int) * img_c);
    }

    print_matrix_hw_img(img_r, img_c, hw_img);

    convolve(
            img_r,
            img_c,
            img,
            kernel_r,
            kernel_c,
            kernel,
            1,
            result
    );


/*
    convolve_offset(
        img_r,
        img_c,
        img,
        kernel_r,
        kernel_c,
        kernel,
        2,
        result
    );
*/
    print_matrix(img_r/input_stride, img_c/input_stride, result);


#ifdef FPGA
    nick_test_conv(hw_img, kernel, hw_res);
#endif

    print_matrix_hw_res(8, 8, hw_res);
    check_first_conv_result(8, 8, result, hw_res);

    free(img);
    free(kernel);
    free(result);
}


/*int main(int argc, char *argv[]) {
    FILE *img_file, *kernel_file;
    size_t img_r, img_c, kernel_r, kernel_c = 0;
    int input_stride = 1;
    img_file = fopen(argv[1], "r");
    kernel_file = fopen(argv[2], "r");

    if (img_file == NULL || kernel_file == NULL){
        perror("Error: ");
        return -1;
    }

    fscanf(img_file, "%d", &img_r);
    fscanf(img_file, "%d", &img_c);
    fscanf(kernel_file, "%d", &kernel_r);
    fscanf(kernel_file, "%d", &kernel_c);

    int** img = read_matrix(img_r, img_c, img_file);
    int** kernel = read_matrix(kernel_r, kernel_c, kernel_file);

    run(img, kernel, img_r, img_c, kernel_r, kernel_c);

    return 0;
}*/
