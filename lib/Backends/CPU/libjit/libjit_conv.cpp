/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <assert.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "libjit_defs.h"

/*extern void glow_conv(int8_t *result, const int8_t *inW, const int8_t *filterW,
                      const int32_t *biasW, const uint32_t *outWdims,
                      const uint32_t *inWdims, const uint32_t *filterWdims,
                      const uint32_t *biasWdims, int32_t outOffset, int32_t inOffset,
                      int32_t filterOffset, int32_t biasOffset, int32_t biasPre,
                      int32_t biasPost, int32_t biasScale, int32_t outPre,
                      int32_t outPost, int32_t outScale);*/

extern void glow_conv(int8_t *result, const int8_t *inW, const int8_t *filterW, const int32_t *biasW, const uint8_t *outWdims, const uint8_t *inWdims,
                      const uint8_t *filterWdims, const uint8_t *biasWdims, int32_t outOffset, int32_t inOffset, int32_t filterOffset,
                      int32_t biasOffset, int32_t biasPre, int32_t biasPost, int32_t biasScale, int32_t outPre, int32_t outPost, int32_t outScale);

namespace {
// Initialize the convolution output frame for slice \p N with the bias \p
// biasW.
void libjit_conv_init_output_with_bias(dim_t N, float *outW, const float *biasW, const dim_t *outWdims, const dim_t *biasWdims) {
    // For each (x,y) step in the output tensor:
    for (dim_t ax = 0; ax < outWdims[1]; ax++) {
        for (dim_t ay = 0; ay < outWdims[2]; ay++) {
            // For each output channel:
            for (dim_t d = 0; d < outWdims[3]; d++) {
                // Store the results to the output buffer.
                float bias = biasW[d];
                auto outIdx = libjit_getXYZW(outWdims, N, ax, ay, d);
                outW[outIdx] = bias;
            } // For each depth in the output.
        }     // For each Y in the output.
    }         // For each X in the output.
}

/// Perform the heart of the convolution. Load \p ywidth scalars in a specific
/// channel, broadcast them, and multiply them with
/// [ywidth * float8 * numDepthRegs] depth values and accumulate them to create
/// [ywidth * float8 * numDepthRegs] depth result values.
void libjit_convDKKC8_convolve_channel(float *outW, const float *inW, const float *filterW, const dim_t *outWdims, const dim_t *inWdims,
                                       const dim_t *filterWdims, size_t sampleN, dim_t outChannel, unsigned numDepthRegs, unsigned ywidth,
                                       dim_t numChannels, sdim_t inX, sdim_t inY, sdim_t outX, sdim_t outY, size_t filterX, size_t filterY,
                                       size_t stride, size_t group) {

    // Process N * YWidth * 8 output pixels at once. Each value here is a
    // scalar that represents the sum for (x,y..y+ywidth) and the filter. The
    // SIMD dimension represents multiple layers of the depth
    // (output channel).
    float8 sum[numDepthRegs][ywidth];
    for (unsigned wu = 0; wu < ywidth; wu++) {
        for (unsigned du = 0; du < numDepthRegs; du++) {
            sum[du][wu] = BroadcastFloat8(0.0);
        }
    }

    // Perform the heart of the convolution.

    // For each input channel:
    for (size_t fd = 0; fd < numChannels; fd++) {
        // First, load and broadcast the scalar data from the input buffer.
        float8 in8[ywidth];
        for (unsigned wu = 0; wu < ywidth; wu++) {
            // Load a single pixel from the input image and broadcast it.
            auto inIdx = libjit_getXYZW(inWdims, sampleN, inX, inY + wu * stride, fd + group * numChannels);
            in8[wu] = BroadcastFloat8(inW[inIdx]);
        }

        // For each y pixel:
        for (unsigned wu = 0; wu < ywidth; wu++) {
            // Load N x 8 elements from the filter layer. The filter is
            // pre-swizzled to ensure efficient access.
            for (unsigned du = 0; du < numDepthRegs; du++) {
                auto filterIdx = libjit_getXYZWQ(filterWdims, outChannel / 8 + du, filterX, filterY, fd, 0);
                float8 ff0 = LoadFloat8(&filterW[filterIdx]);
                sum[du][wu] += ff0 * in8[wu];
            }
        }
    }

    // Store the results to the output buffer.
    for (unsigned wu = 0; wu < ywidth; wu++) {
        for (unsigned du = 0; du < numDepthRegs; du++) {
            // Add the partial sum to the tile.
            auto outIdx = libjit_getXYZW(outWdims, sampleN, outX, outY + wu, outChannel + du * 8);
            AddFloat8(&outW[outIdx], sum[du][wu]);
        }
    }
}

/// Process the input buffer in the convolution by iterating on the filter and
/// then on the pixels. This means that we process the whole input image for
/// each pixel in the filter. We try to unroll and process multiple inputs on
/// the Y row together.
void libjit_convDKKC8_foreach_xy_filter_pixels(size_t sampleN, dim_t outChannel, unsigned numDepthRegs, unsigned depthStrips, unsigned sizeGroupY,
                                               dim_t numChannels, float *outW, const float *inW, const float *filterW, const float *biasW,
                                               const dim_t *outWdims, const dim_t *inWdims, const dim_t *filterWdims, const dim_t *biasWdims,
                                               const dim_t *kernelSizes, const dim_t *strides, const dim_t *pads, dim_t group,
                                               dim_t endChannelIndex) {
    // The loops below look scary but the idea is simple. We iterate over
    // the pixels in the output tensor and calculate the coordinate of the
    // source tensor. When we process the Y row we try to process [sizeGroupY]
    // elements at once. After we finish the row we handle the odd cases by
    // handling one y value at a time.

    dim_t pad_t = pads[0];
    dim_t pad_l = pads[1];
    dim_t stride_h = strides[0];
    dim_t stride_w = strides[1];
    dim_t kernel_h = kernelSizes[0];
    dim_t kernel_w = kernelSizes[1];
    // For each element in the convolution-filter:
    for (dim_t fx = 0; fx < kernel_h; fx++) {
        for (dim_t fy = 0; fy < kernel_w; fy++) {

            // For each x step in the input/output tensor:
            for (dim_t outx = 0; outx < outWdims[1]; outx++) {
                sdim_t inx = (sdim_t)outx * stride_h - pad_t + fx;

                // Ignore out-of-bounds X values.
                if (inx < 0 || inx >= (sdim_t)inWdims[1]) {
                    continue;
                }

                // For each y step in the input/output tensor, in steps of \p
                // sizeGroupY. We process \p sizeGroupY pixels of Y in one
                // iteration.
                dim_t outy = 0;
                while (outy < outWdims[2]) {
                    sdim_t iny = (sdim_t)outy * stride_w - pad_l + fy;

                    if ((sdim_t)(iny + (sdim_t)stride_w * sizeGroupY) >= (sdim_t)inWdims[2]) {
                        // If we've passed the upper bound, we don't want to
                        // increment `outy` again, since we're going to handle
                        // the remaining y steps in the following loop.
                        break;
                    }
                    // Ignore out of bound indices.
                    if (iny < 0) {
                        /// We know iny is out of bounds, so we have nothing to
                        /// do for outy. But we can't skip ahead by sizeGroupY,
                        /// because we haven't checked outy + 1.
                        outy += 1;
                        continue;
                    }

                    // Convolve the (x,y .. y + ywidth) values.
                    dim_t outC = outChannel;
                    for (unsigned strip = 0; strip < depthStrips && outC < endChannelIndex; strip++) {
                        libjit_convDKKC8_convolve_channel(outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN, outC, numDepthRegs, sizeGroupY,
                                                          numChannels, inx, iny, outx, outy, fx, fy, stride_w, group);
                        outC += numDepthRegs * 8;
                    }

                    outy += sizeGroupY;
                } // For each Y group in the output.

                // Handle the remaining Y in the row in groups of size 1.
                for (; outy < outWdims[2]; outy++) {
                    sdim_t iny = (sdim_t)outy * stride_w - pad_l + fy;
                    // Ignore out of bound indices.
                    if (iny < 0 || iny >= (sdim_t)inWdims[2]) {
                        continue;
                    }

                    // Convolve a single (x,y) value.
                    dim_t outC = outChannel;
                    for (unsigned strip = 0; strip < depthStrips && outC < endChannelIndex; strip++) {
                        libjit_convDKKC8_convolve_channel(outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN, outC, numDepthRegs, 1,
                                                          numChannels, inx, iny, outx, outy, fx, fy, stride_w, group);
                        outC += numDepthRegs * 8;
                    }
                } // For each Y, in step of 1, in the output.

            } // For each X in the output.
        }     // For each Y in the filter.
    }         // For each X in the filter.
}

// Process the input buffer in the convolution by iterating on the input buffer
// and then on the filter. This means that we process the whole input filter for
// each pixel in the input buffer.
void libjit_convDKKC8_foreach_xy_pixels_filter(size_t sampleN, dim_t outChannel, unsigned numDepthRegs, unsigned depthStrips, unsigned sizeGroupY,
                                               dim_t numChannels, float *outW, const float *inW, const float *filterW, const float *biasW,
                                               const dim_t *outWdims, const dim_t *inWdims, const dim_t *filterWdims, const dim_t *biasWdims,
                                               const dim_t *kernelSizes, const dim_t *strides, const dim_t *pads, dim_t group,
                                               dim_t endChannelIndex) {

    dim_t pad_t = pads[0];
    dim_t pad_l = pads[1];
    dim_t stride_h = strides[0];
    dim_t stride_w = strides[1];
    dim_t kernel_h = kernelSizes[0];
    dim_t kernel_w = kernelSizes[1];
    // For each (x,y) step in the input/output tensor:
    for (dim_t outx = 0; outx < outWdims[1]; outx++) {
        for (dim_t outy = 0; outy < outWdims[2]; outy++) {

            // For each element in the convolution-filter:
            for (dim_t fx = 0; fx < kernel_h; fx++) {
                for (dim_t fy = 0; fy < kernel_w; fy++) {

                    // Calculate the specific input x,y that we process in this
                    // iteration.
                    dim_t inx = (dim_t)outx * stride_h - pad_t + fx;
                    dim_t iny = (dim_t)outy * stride_w - pad_l + fy;

                    // Ignore index access below zero (this is due to padding).
                    if (inx < 0 || iny < 0 || inx >= inWdims[1] || iny >= inWdims[2]) {
                        continue;
                    }

                    dim_t outC = outChannel;
                    for (unsigned strip = 0; strip < depthStrips && outC < endChannelIndex; strip++) {
                        libjit_convDKKC8_convolve_channel(outW, inW, filterW, outWdims, inWdims, filterWdims, sampleN, outC, numDepthRegs, 1,
                                                          numChannels, inx, iny, outx, outy, fx, fy, stride_w, group);
                        outC += numDepthRegs * 8;
                    }
                } // For each Y in the filter.
            }     // For each X in the filter.
        }         // For each Y in the output.
    }             // For each X in the output.
}

// ** Our print matrix
#define debug 1
#ifdef debug

void print_simple_matrix(dim_t rows, dim_t cols, const signed char *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j < cols - 1)
                printf("%04d ", matrix[i * rows + j]);
            else
                printf("%04d", matrix[i * rows + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_simple_matrix_32(dim_t rows, dim_t cols, const signed int *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j < cols - 1)
                printf("%04d ", matrix[i * rows + j]);
            else
                printf("%04d", matrix[i * rows + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrix(dim_t rows, dim_t cols, const signed char *matrix) {
    for (int k = 0; k < rows * rows * rows; k += rows * rows) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (j < cols - 1)
                    printf("%04d ", matrix[k + i * rows + j]);
                else
                    printf("%04d", matrix[k + i * rows + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_layer_output(dim_t rows, dim_t cols, dim_t channels, const signed char *matrix) {
    for (int j = 0; j < channels; j++) {
        for (int k = 0; k < rows * cols; k += cols) {
            for (int i = 0; i < rows * cols; i += rows) {
                printf("%04d ", matrix[j + (k * rows) + i]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void row_write_layer_output(dim_t rows, dim_t cols, dim_t channels, const signed char *matrix) {

    FILE *layer_output_file = fopen("result_output.txt", "w");

    for (int n = 0; n < channels; n++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fprintf(layer_output_file, "%04d ", matrix[(n * channels * rows) + i * cols + j]);
            }
            fprintf(layer_output_file, "\n");
        }
        fprintf(layer_output_file, "\n");
    }

    fclose(layer_output_file);
}

void column_write_layer_output(dim_t rows, dim_t cols, dim_t channels, const signed char *matrix) {
    char buf[21];

    snprintf(buf, 21, "layer_%03llu_output.txt", channels);

    FILE *layer_output_file = fopen(buf, "w");

    for (int j = 0; j < channels; j++) {
        for (int k = 0; k < rows * cols; k += cols) {
            for (int i = 0; i < rows * cols; i += rows) {
                fprintf(layer_output_file, "%04d ", matrix[j + (k * rows) + i]);
            }
            fprintf(layer_output_file, "\n");
        }
        fprintf(layer_output_file, "\n");
    }

    fclose(layer_output_file);
}
#endif // debug

template <typename ElemTy, typename BiasElemTy>
void dlha_layer1_conv(ElemTy *outW, const ElemTy *inW, const ElemTy *filterW, const BiasElemTy *biasW, const dim_t *outWdims, const dim_t *inWdims,
                      const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes, const dim_t *strides, const dim_t *pads,
                      dim_t group, int32_t outOffset, int32_t inOffset, int32_t filterOffset, int32_t biasOffset, int32_t biasPre, int32_t biasPost,
                      int32_t biasScale, int32_t outPre, int32_t outPost, int32_t outScale, unsigned depthUnroll, dim_t dilation) {
    dim_t inChannels = inWdims[3];
    dim_t outChannels = outWdims[3];
    dim_t inCperG = inChannels / group;
    dim_t outCperG = outChannels / group;
    dim_t pad_t = pads[0];
    dim_t pad_l = pads[1];
    dim_t stride_h = strides[0];
    size_t stride_w = strides[1];
    size_t kernel_h = kernelSizes[0];
    size_t kernel_w = kernelSizes[1];

    // size_t == lu == unsigned long
    // dim_t == llu = unsigned long long

    /*    FILE *bias_output_file = fopen("bias.txt", "w");
        fprintf(bias_output_file, "32 32\n\n");
        for (int j = 0; j < 32; j++) {
            for (int i = 0; i < 32; i++) {
                fprintf(bias_output_file, "%d ", libjit_scale_i32i8((int32_t)biasW[0] - biasOffset, biasPre, biasPost, biasScale, 0));
            }
            fprintf(bias_output_file, "\n");
        }
        fclose(bias_output_file);

        FILE *img_output_file = fopen("img.txt", "w");
        fprintf(bias_output_file, "32 32\n\n");
        for (int j = 0; j < 32; j++) {
            for (int i = 0; i < 32; i++) {
                fprintf(img_output_file, "%d ", inW[j * 32 + i]);
            }
            fprintf(img_output_file, "\n");
        }
        fclose(img_output_file);

        FILE *kernel_output_file = fopen("kernel.txt", "w");
        fprintf(kernel_output_file, "3 3\n\n");
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
                fprintf(kernel_output_file, "%d ", filterW[j * 3 + i]);
            }
            fprintf(kernel_output_file, "\n");
        }
        fclose(kernel_output_file);

        FILE *offset_output_file = fopen("offset.txt", "w");
        fprintf(offset_output_file, "%d ", inOffset);
        fprintf(offset_output_file, "%d", filterOffset);
        fclose(offset_output_file);*/

    const uint8_t small_inWdims[4] = {(uint8_t)inWdims[0], (uint8_t)inWdims[1], (uint8_t)inWdims[2], (uint8_t)inWdims[3]};
    const uint8_t small_filterWdims[4] = {(uint8_t)filterWdims[0], (uint8_t)filterWdims[1], (uint8_t)filterWdims[2], (uint8_t)filterWdims[3]};
    const uint8_t small_biasWdims[1] = {(uint8_t)biasWdims[0]};
    const uint8_t small_outWdims[4] = {(uint8_t)outWdims[0], (uint8_t)outWdims[1], (uint8_t)outWdims[2], (uint8_t)outWdims[3]};

#ifdef debug
    printf("small_inWdims: [%u,%u,%u,%u]\n", small_inWdims[0], small_inWdims[1], small_inWdims[2], small_inWdims[3]);
    printf("small_filterWdims: [%u,%u,%u,%u]\n", small_filterWdims[0], small_filterWdims[1], small_filterWdims[2], small_filterWdims[3]);
    printf("small_biasWdims: [%u]\n", small_biasWdims[0]);
    printf("biasW: ");
    for (int i = 0; i < biasWdims[0]; i++) {
        printf("%d,", (int8_t)libjit_scale_i32i8((int32_t)biasW[i] - biasOffset, biasPre, biasPost, biasScale, 0));
    }
    printf("\n");
    printf("small_outWdims: [%u,%u,%u,%u]\n", small_outWdims[0], small_outWdims[1], small_outWdims[2], small_outWdims[3]);

    printf("group: %llu\n", group);             // always 0
    printf("inOffset: %d\n", inOffset);         // -128 > -106 > -60
    printf("depthUnroll: %u\n", depthUnroll);   // always 8
    printf("inChannels: %llu\n", inChannels);   // 1 > 32 >  64
    printf("outChannels: %llu\n", outChannels); // 32 > 64 > 128
    printf("inCperG: %llu\n", inCperG);         // 1 > 32 >  64
    printf("outCperG: %llu\n", outCperG);       // 32 > 64 > 128
    printf("pad_t: %llu\n", pad_t);             // always 1
    printf("pad_l: %llu\n", pad_l);             // always 1
    printf("dilation: %llu\n", dilation);       // always 1

    printf("filterOffset: %d\n", filterOffset);

    printf("biasWdims[0]: %llu\n", biasWdims[0]);

    printf("biasOffset: %d\n", biasOffset); // 0
    printf("biasPre: %d\n", biasPre);       // 0
    printf("biasPost: %d\n", biasPost);     // 0
    printf("biasScale: %d\n", biasScale);   // 1

    printf("outOffset: %d\n", outOffset); // -22
    printf("outPre: %d\n", outPre);       // 3
    printf("outPost: %d\n", outPost);     // 15
    printf("outScale: %d\n", outScale);   // 300
#endif                                    // debug

    depthUnroll = 1;

    /*    int32_t bias[biasWdims[0] * biasWdims[0]];

        for (int y = 0; y < inWdims[1]; y += 1) {
            for (int x = 0; x < inWdims[2]; x += 1) {
                bias[y * 32 + x] = libjit_scale_i32i8((int32_t)biasW[0] - biasOffset, biasPre, biasPost, biasScale, 0);
            }
        }

        print_simple_matrix_32(outWdims[1], outWdims[2], bias);

        int32_t img[inWdims[1] * inWdims[2]];

        for (int y = 0; y < inWdims[1]; y += 1) {
            for (int x = 0; x < inWdims[2]; x += 1) {
                img[y * 32 + x] = inW[y * 32 + x];
            }
        }
        print_simple_matrix_32(outWdims[1], outWdims[2], img);

        int32_t filter[filterWdims[1] * filterWdims[2]];

        for (int y = 0; y < filterWdims[1]; y += 1) {
            for (int x = 0; x < filterWdims[2]; x += 1) {
                filter[y * 3 + x] = filterW[y * 3 + x];
            }
        }
        print_simple_matrix_32(filterWdims[1], filterWdims[2], filter);*/

    int8_t result[small_outWdims[0] * small_outWdims[1] * small_outWdims[2] * small_outWdims[3]];

    /*    glow_conv(result, inW, filterW, biasW, (uint32_t *) outWdims, (uint32_t *) inWdims, (uint32_t *) filterWdims, (uint32_t *) biasWdims,
       outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale);*/
    glow_conv(result, inW, filterW, biasW, small_outWdims, small_inWdims, small_filterWdims, small_biasWdims, outOffset, inOffset, filterOffset,
              biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale);

    //    row_write_layer_output(outWdims[1], outWdims[2], outWdims[3], result);

    FILE *first_hardware_outW = fopen("first_hardware_outW.txt", "w");
    for (int i = 0; i < 32 * 32 * 32; i++)
        fprintf(first_hardware_outW, "%d,", result[i]);
    fprintf(first_hardware_outW, "\n");
    fclose(first_hardware_outW);

    //    for (size_t n = 0; n < inWdims[0]; n++) {

    for (size_t n = 0; n < outCperG; n += depthUnroll) {

        for (size_t j = 0; j < outWdims[1]; j++) {

            for (size_t i = 0; i < outWdims[2]; i++) {

                outW[libjit_getXYZW(outWdims, 0, j, i, n)] = result[(n * outWdims[1] * outWdims[2]) + j * outWdims[1] + i];
            }
        }
    }
    //    }

    //    size_t g = 0;

    //    FILE *kernel_file = fopen("kernel_output.txt", "w");
    /*    FILE *res_output_file = fopen("res.txt", "w");
        fprintf(res_output_file, "32 32\n\n");

        int jump = 0;

        // For each input in the batch:
        for (size_t n = 0; n < inChannels; n++) { // n: 0
            // For each output channel in the group. Process 'depthUnroll' output layers together.
            for (size_t d = 0; d < outCperG; d += depthUnroll) { // d: 0 -> 8 -> 16 -> 24 // outCperG
                // For each convolution 'jump' in the input tensor:
                ssize_t x = -(ssize_t)pad_t; // -1 -> 30

                for (size_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) { // 32

                    ssize_t y = -(ssize_t)pad_l;

                    for (size_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) { // 32
                        int32_t sum;                                             // int32_t sum[depthUnroll];

                        sum = libjit_scale_i32i8((int32_t)biasW[d] - biasOffset, biasPre, biasPost, biasScale, 0);

                        // For each element in the convolution-filter:
                        for (size_t fx = 0; fx < kernel_h; fx++) {     // 0, 1, 2
                            for (size_t fy = 0; fy < kernel_w; fy++) { // 0, 1, 2
                                ssize_t ox = x + fx;                   // * dilation; ox: -1 -> 32
                                ssize_t oy = y + fy;                   // * dilation; oy: -1 -> 32

                                // Ignore index access below zero (this is due to padding).
                                if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] || oy >= (ssize_t)inWdims[2]) {
                                    continue;
                                }

                                // Calculate the indices into the Filter and Input buffers.
                                //                                size_t inIdx = libjit_getXYZW(inWdims, n, (size_t) ox, (size_t) oy, g * inCperG);
                                size_t inIdx = (n * inWdims[1] * inWdims[2] * inWdims[3]) + // 0 * 32 * 32 * 1
                                               (ox * inWdims[2] * inWdims[3]) +             // (0 -> 31) * 32 * 1
                                               (oy * inWdims[3]) +                          // (0 -> 31) * 1
                                               (g * inCperG);                               // 0 * 1

                                //                            printf("%lu ", inIdx);

                                //                                size_t filterIdx = libjit_getXYZW(filterWdims, d, fx, fy, 0);
                                size_t filterIdx = (d * filterWdims[1] * filterWdims[2] * filterWdims[3]) + (fx * filterWdims[2] * filterWdims[3]) +
                                                   (fy * filterWdims[3]);

                                //                                printf("%lu,", filterIdx);

                                //                            size_t sliceSize = filterWdims[1] * filterWdims[2] * filterWdims[3]; // 3 * 3 * 1

                                for (size_t fd = 0; fd < inCperG; fd++) { // 0
                                    int32_t in = inW[inIdx + fd] - inOffset;
                                    //                                printf("%d,", inW[inIdx + fd] - inOffset);
                                    sum += (filterW[filterIdx + fd] - filterOffset) * in;
                                }
                            }
                        }

                        if (d == 0) {
                            //                        if (jump % 32 == 0) fprintf(res_output_file, "\n");
                            //                        if (jump % 1024 == 0) fprintf(res_output_file, "\n");
                            //                    printf("%lu,", libjit_getXYZW(outWdims, n, ax, ay, d));
                            fprintf(res_output_file, "%d ", sum);
                            jump++;
                        }

                        int32_t scaledSum = libjit_scale_i32i8(sum, outPre, outPost, outScale, outOffset);

                        outW[libjit_getXYZW(outWdims, n, ax, ay, d)] = libjit_clip(scaledSum);
                    } // W
                    fprintf(res_output_file, "\n");
                } // H
            }     // C
        }         // N
                  //    fclose(kernel_file);
        fclose(res_output_file);*/
    /*#ifdef debug
        printf("\n********************** PRINTING OUTPUT IMAGE(s): AFTER **************************\n");
        write_layer_output(outWdims[1], outWdims[2], outCperG, outW);
        print_layer_output(outWdims[1], outWdims[2], outCperG, outW);
        printf("[%d,%d,%d,]", outWdims[1], outWdims[2], outCperG);
    #endif // debug*/
}

template <typename ElemTy, typename BiasElemTy>
void dlha_conv(ElemTy *outW, const ElemTy *inW, const ElemTy *filterW, const BiasElemTy *biasW, const dim_t *outWdims, const dim_t *inWdims,
               const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes, const dim_t *strides, const dim_t *pads, dim_t group,
               int32_t outOffset, int32_t inOffset, int32_t filterOffset, int32_t biasOffset, int32_t biasPre, int32_t biasPost, int32_t biasScale,
               int32_t outPre, int32_t outPost, int32_t outScale, unsigned depthUnroll, dim_t dilation) {
    //    dim_t inChannels = inWdims[3];
    //    dim_t outChannels = outWdims[3];
    //    dim_t inCperG = inChannels / group;
    //    dim_t outCperG = outChannels / group;
    //    dim_t pad_t = pads[0];
    //    dim_t pad_l = pads[1];
    //    size_t kernel_h = kernelSizes[0];
    //    size_t kernel_w = kernelSizes[1];

    // size_t == lu == unsigned long
    // dim_t == llu = unsigned long long

#ifdef debug
//    printf("small_inWdims: [%u,%u,%u,%u]\n", small_inWdims[0], small_inWdims[1], small_inWdims[2], small_inWdims[3]);
//    printf("small_filterWdims: [%u,%u,%u,%u]\n", small_filterWdims[0], small_filterWdims[1], small_filterWdims[2], small_filterWdims[3]);
//
//    printf("small_biasWdims: [%u]\n", small_biasWdims[0]);
//    printf("biasW: ");
//    for (int i = 0; i < biasWdims[0]; i++) {
//        printf("%d,", (int8_t)libjit_scale_i32i8((int32_t)biasW[i] - biasOffset, biasPre, biasPost, biasScale, 0));
//    }
//    printf("\n");
//    printf("small_outWdims: [%u,%u,%u,%u]\n", small_outWdims[0], small_outWdims[1], small_outWdims[2], small_outWdims[3]);

//    printf("group: %llu\n", group);             // always 0
//    printf("inOffset: %d\n", inOffset);         // -128 > -106 > -60
//    printf("depthUnroll: %u\n", depthUnroll);   // always 8
//    printf("inChannels: %llu\n", inChannels);   // 1 > 32 >  64
//    printf("outChannels: %llu\n", outChannels); // 32 > 64 > 128

//    printf("pad_t: %llu\n", pad_t);             // always 1
//    printf("pad_l: %llu\n", pad_l);             // always 1
//    printf("dilation: %llu\n", dilation);       // always 1

//    printf("filterOffset: %d\n", filterOffset);

/*    printf("biasOffset: %d\n", biasOffset); // 0
    printf("biasPre: %d\n", biasPre);       // 0
    printf("biasPost: %d\n", biasPost);     // 0
    printf("biasScale: %d\n", biasScale);   // 1

    printf("outOffset: %d\n", outOffset); // -22
    printf("outPre: %d\n", outPre);       // 3
    printf("outPost: %d\n", outPost);     // 15
    printf("outScale: %d\n", outScale);   // 300*/
#endif

    const uint8_t small_inWdims[4] = {(uint8_t)inWdims[0], (uint8_t)inWdims[1], (uint8_t)inWdims[2], (uint8_t)inWdims[3]};
    const uint8_t small_filterWdims[4] = {(uint8_t)filterWdims[0], (uint8_t)filterWdims[1], (uint8_t)filterWdims[2], (uint8_t)filterWdims[3]};
    const uint8_t small_biasWdims[1] = {(uint8_t)biasWdims[0]};
    const uint8_t small_outWdims[4] = {(uint8_t)outWdims[0], (uint8_t)outWdims[1], (uint8_t)outWdims[2], (uint8_t)outWdims[3]};

    glow_conv(outW, inW, filterW, biasW, small_outWdims, small_inWdims, small_filterWdims, small_biasWdims, outOffset, inOffset, filterOffset,
              biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale);

    /*    // For each output channel
        for (size_t c = 0; c < outWdims[3]; c++) {
            // For each convolution 'jump' in the input tensor:
            ssize_t x = -(ssize_t) 1;
            for (size_t ax = 0; ax < outWdims[1]; x++, ax++) {
                ssize_t y = -(ssize_t) 1;
                for (size_t ay = 0; ay < outWdims[2]; y++, ay++) {
                    int32_t sum = libjit_scale_i32i8((int32_t) biasW[c] - biasOffset, biasPre, biasPost, biasScale, 0);

                    // For each element in the convolution-filter:
                    for (size_t fx = 0; fx < kernel_h; fx++) {
                        for (size_t fy = 0; fy < kernel_w; fy++) {
                            ssize_t ox = x + fx; // 0 -> 15
                            ssize_t oy = y + fy; // 0 -> 15

                            // Ignore index access below zero (this is due to padding).
                            if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] || oy >= (ssize_t) inWdims[2]) {
                                continue;
                            }

    //                        inline dim_t libjit_getXYZW(const dim_t *dims, dim_t x, dim_t y, dim_t z, dim_t w)
    //                            return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) + (z * dims[3]) + w;

                            // inIdx = (ox * 16 * 32) + (oy * 32)

                            // inIdx: 0,32,64,96

                            // Calculate the indices into the Filter and Input buffers.
                            size_t inIdx = libjit_getXYZW(inWdims, 0, (size_t) ox, (size_t) oy, 0); // 0 -> 8160
                            size_t filterIdx = libjit_getXYZW(filterWdims, c, fx, fy, 0);

                            for (size_t fd = 0; fd < inCperG; fd++) { // 0 -> 31
                                int32_t in = inW[inIdx + fd] - inOffset;
                                sum += (filterW[filterIdx + fd] - filterOffset) * in;
                            }
                        }
                    }

                    int32_t scaledSum = libjit_scale_i32i8(sum, outPre, outPost, outScale, outOffset);
                    outW[libjit_getXYZW(outWdims, 0, ax, ay, c)] = libjit_clip(scaledSum);
                } // W
            }     // H
        }         // C*/
}

/// Generic template for quantized convolution. The template allows choosing
/// element type and bias type.
template <typename ElemTy, typename BiasElemTy>
void libjit_quantized_convolution_generic(ElemTy *outW, const ElemTy *inW, const ElemTy *filterW, const BiasElemTy *biasW, const dim_t *outWdims,
                                          const dim_t *inWdims, const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes,
                                          const dim_t *strides, const dim_t *pads, dim_t group, int32_t outOffset, int32_t inOffset,
                                          int32_t filterOffset, int32_t biasOffset, int32_t biasPre, int32_t biasPost, int32_t biasScale,
                                          int32_t outPre, int32_t outPost, int32_t outScale, unsigned depthUnroll, dim_t dilation) {

    dim_t inChannels = inWdims[3];
    dim_t outChannels = outWdims[3];
    dim_t inCperG = inChannels / group;
    dim_t outCperG = outChannels / group;
    dim_t pad_t = pads[0];
    dim_t pad_l = pads[1];
    dim_t stride_h = strides[0];
    size_t stride_w = strides[1];
    size_t kernel_h = kernelSizes[0];
    size_t kernel_w = kernelSizes[1];
    // For each input in the batch:
    for (size_t n = 0; n < inWdims[0]; n++) {
        // For each group of input channels:
        for (size_t g = 0; g < group; g++) {

            // For each output channel in the group. Process 'depthUnroll' output
            // layers together.
            for (size_t d = g * outCperG; d < (g + 1) * outCperG; d += depthUnroll) {
                // For each convolution 'jump' in the input tensor:
                ssize_t x = -(ssize_t)pad_t;
                for (size_t ax = 0; ax < outWdims[1]; x += stride_h, ax++) {
                    ssize_t y = -(ssize_t)pad_l;
                    for (size_t ay = 0; ay < outWdims[2]; y += stride_w, ay++) {
                        int32_t sum[depthUnroll];

                        for (unsigned i = 0; i < depthUnroll; i++) {
                            // Scale the bias to match the scale of the matrix multiplication.
                            sum[i] = libjit_scale_i32i8((int32_t)biasW[d + i] - biasOffset, biasPre, biasPost, biasScale, 0);
                        }

                        // For each element in the convolution-filter:
                        for (size_t fx = 0; fx < kernel_h; fx++) {
                            for (size_t fy = 0; fy < kernel_w; fy++) {
                                ssize_t ox = x + fx * dilation;
                                ssize_t oy = y + fy * dilation;

                                // Ignore index access below zero (this is due to padding).
                                if (ox < 0 || oy < 0 || ox >= (ssize_t)inWdims[1] || oy >= (ssize_t)inWdims[2]) {
                                    continue;
                                }

                                // Calculate the indices into the Filter and Input buffers.
                                size_t inIdx = libjit_getXYZW(inWdims, n, (size_t)ox, (size_t)oy, g * inCperG);
                                size_t filterIdx = libjit_getXYZW(filterWdims, d, fx, fy, 0);
                                size_t sliceSize = filterWdims[1] * filterWdims[2] * filterWdims[3];

                                // Perform the innermost loop of the convolution using 4 vector
                                // registers.
                                for (size_t fd = 0; fd < inCperG; fd++) {
                                    int32_t in = inW[inIdx + fd] - inOffset;
                                    for (unsigned i = 0; i < MIN(4, depthUnroll); i++) {
                                        sum[i] += (filterW[filterIdx + (sliceSize * i) + fd] - filterOffset) * in;
                                    }
                                }

                                // And perform the innermost loop again with 4 more registers.
                                if (depthUnroll > 4)
                                    for (size_t fd = 0; fd < inCperG; fd++) {
                                        int32_t in = inW[inIdx + fd] - inOffset;
                                        for (unsigned i = 4; i < MIN(8, depthUnroll); i++) {
                                            sum[i] += (filterW[filterIdx + (sliceSize * i) + fd] - filterOffset) * in;
                                        }
                                    }
                            }
                        }

                        for (unsigned i = 0; i < depthUnroll; i++) {
                            // Scale the result back to the expected destination scale.
                            int32_t scaledSum = libjit_scale_i32i8(sum[i], outPre, outPost, outScale, outOffset);
                            outW[libjit_getXYZW(outWdims, n, ax, ay, d + i)] = libjit_clip(scaledSum);
                        }
                    } // W
                }     // H
            }         // C
        }             // G
    }                 // N
}

} // namespace

extern "C" {
void libjit_convDKKC8_f(float *outW, const float *inW, const float *filterW, const float *biasW, const dim_t *outWdims, const dim_t *inWdims,
                        const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes, const dim_t *strides, const dim_t *pads,
                        dim_t group, unsigned pixelScanFirst, unsigned numDepthRegs, unsigned sizeGroupY, unsigned depthStrips) {
    dim_t inChannels = inWdims[3];
    dim_t outChannels = outWdims[3];
    dim_t inCperG = inChannels / group;
    dim_t outCperG = outChannels / group;

    // Select the order in which we iterate over the pixels in the picture.
    auto eachPixelConv = (pixelScanFirst ? &libjit_convDKKC8_foreach_xy_pixels_filter : &libjit_convDKKC8_foreach_xy_filter_pixels);

    // For each input in the batch:
    for (dim_t n = 0; n < inWdims[0]; n++) {

        // Initialize the output frame for the N'th slice with the bias.
        // Later we will accumulate values into this slice.
        libjit_conv_init_output_with_bias(n, outW, biasW, outWdims, biasWdims);

        // For each group of input channels:
        for (dim_t g = 0; g < group; g++) {

            // For each output channel, process [numDepthRegs x float8]
            // elements.
            dim_t startChannelIndex = g * outCperG;
            dim_t endChannelIndex = (g + 1) * outCperG;
            for (dim_t d = startChannelIndex; d < endChannelIndex; d += 8 * numDepthRegs * depthStrips) {

                // Perform the convolution for each pixel.
                eachPixelConv(n, d, numDepthRegs, depthStrips, sizeGroupY, inCperG, outW, inW, filterW, biasW, outWdims, inWdims, filterWdims,
                              biasWdims, kernelSizes, strides, pads, g, endChannelIndex);

            } // For each D (the depth, or the output channel).
        }     // for each G, the group
    }         // For each N, the sample in the batch.
}

void libjit_convolution_f(float *outW, const float *inW, const float *filterW, const float *biasW, const dim_t *outWdims, const dim_t *inWdims,
                          const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes, const dim_t *strides, const dim_t *pads,
                          dim_t group, unsigned depthUnroll, dim_t dilation) {
    dim_t inChannels = inWdims[3];
    dim_t outChannels = outWdims[3];
    dim_t inCperG = inChannels / group;
    dim_t outCperG = outChannels / group;

    // The output dims are calculated already from all of the pads,
    // therefore we only need the top and left pads here to control the starting
    // position.
    dim_t pad_t = pads[0];
    dim_t pad_l = pads[1];
    dim_t stride_h = strides[0];
    dim_t stride_w = strides[1];
    dim_t kernel_h = kernelSizes[0];
    dim_t kernel_w = kernelSizes[1];
    // The size of the input-channel tile. High channel count allow for SIMD
    // parallelism but create register pressure. Low channel count reduces the
    // memory pressure and allows things to fit in cache, but require additional
    // compute (horizontal add) to sum the values in the block. This value is a
    // compromise between the two.
    constexpr unsigned cbSize = 512;

    // For each input in the batch:
    for (dim_t n = 0; n < inWdims[0]; n++) {

        // Initialize the output frame for the N'th slice with the bias.
        // Later we will accumulate values into this slice.
        libjit_conv_init_output_with_bias(n, outW, biasW, outWdims, biasWdims);

        // For each group of input channels:
        for (dim_t g = 0; g < group; g++) {
            // Process the body of the loop in tiles of "channel-block".
            for (dim_t cb = 0; cb < inCperG; cb += cbSize) {

                // For each output channel in the group. Process 'depthUnroll'
                // output layers together.
                for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d += depthUnroll) {

                    // For each element in the convolution-filter:
                    for (dim_t fx = 0; fx < kernel_h; fx++) {
                        for (dim_t fy = 0; fy < kernel_w; fy++) {

                            // For each convolution 'jump' in the input tensor:
                            for (dim_t outx = 0; outx < outWdims[1]; outx++) {
                                for (dim_t outy = 0; outy < outWdims[2]; outy++) {

                                    // Process 'depthUnroll' output pixels at
                                    // once. Each scalar here represents the
                                    // convolution sum for one (x,y) point in
                                    // the output. We process the same pixel for
                                    // different output channel (D) values. The
                                    // compiler should perform scalar
                                    // replacement of aggregates and split this
                                    // tiny array to registers.
                                    float sum[depthUnroll];
                                    for (unsigned i = 0; i < depthUnroll; i++) {
                                        sum[i] = 0;
                                    }

                                    // Calculate the specific input x,y that we
                                    // process in this iteration.
                                    sdim_t inx = (sdim_t)outx * stride_h - pad_t + fx * dilation;
                                    sdim_t iny = (sdim_t)outy * stride_w - pad_l + fy * dilation;

                                    // Ignore index access below zero (this is
                                    // due to padding).
                                    if (inx < 0 || iny < 0 || inx >= (sdim_t)inWdims[1] || iny >= (sdim_t)inWdims[2]) {
                                        continue;
                                    }

                                    // Calculate the indices into the Filter and
                                    // Input buffers.
                                    dim_t inIdx = libjit_getXYZW(inWdims, n, (dim_t)inx, (dim_t)iny, g * inCperG);
                                    dim_t filterIdx = libjit_getXYZW(filterWdims, d, fx, fy, 0);
                                    dim_t sliceSize = filterWdims[1] * filterWdims[2] * filterWdims[3];

                                    // Perform the heart of the convolution, 4
                                    // elements at a time to reduce register
                                    // pressure.
                                    for (dim_t fd = cb, e = MIN(cb + cbSize, inCperG); fd < e; fd++) {
                                        float in = inW[inIdx + fd];
                                        for (unsigned i = 0; i < MIN(4, depthUnroll); i++) {
                                            sum[i] += filterW[filterIdx + (sliceSize * i) + fd] * in;
                                        }
                                    }

                                    // And run the innermost loop again for the
                                    // second group of depth slices:
                                    if (depthUnroll > 4) {
                                        for (dim_t fd = cb, e = MIN(cb + cbSize, inCperG); fd < e; fd++) {
                                            float in = inW[inIdx + fd];
                                            for (unsigned i = 4; i < MIN(8, depthUnroll); i++) {
                                                sum[i] += filterW[filterIdx + (sliceSize * i) + fd] * in;
                                            }
                                        }
                                    }

                                    // Store the results to the output buffer.
                                    for (unsigned i = 0; i < depthUnroll; i++) {
                                        outW[libjit_getXYZW(outWdims, n, outx, outy, d + i)] += sum[i];
                                    }
                                }
                            }
                        } // For each Y in the filter.
                    }     // For each X in the filter.
                }         // For each D (the depth, or the output channel).
            }             // For each block in the input channel.
        }                 // For each group in the input channel.
    }                     // For each N, the sample in the batch.
}

void libjit_convolution_i8_i32(int8_t *outW, const int8_t *inW, const int8_t *filterW, const int32_t *biasW, const dim_t *outWdims,
                               const dim_t *inWdims, const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes, const dim_t *strides,
                               const dim_t *pads, dim_t group, int32_t outOffset, int32_t inOffset, int32_t filterOffset, int32_t biasOffset,
                               int32_t biasPre, int32_t biasPost, int32_t biasScale, int32_t outPre, int32_t outPost, int32_t outScale,
                               unsigned depthUnroll, dim_t dilation) {
    printf("JOST IN libjit_convolution_i8_i32\n");

    dlha_conv<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes, strides, pads, group, outOffset,
                               inOffset, filterOffset, biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale, depthUnroll,
                               dilation);

/*    char hardware_buf[20];
    snprintf(hardware_buf, 20, "hardware_outW_%02llu.txt", inWdims[3]);

    FILE *hardware_outW = fopen(hardware_buf, "w");
    for (int i = 0; i < outWdims[0] * outWdims[1] * outWdims[2] * outWdims[3]; i++)
        fprintf(hardware_outW, "%d,", outW[i]);
    fprintf(hardware_outW, "\n");
    fclose(hardware_outW);*/

    libjit_quantized_convolution_generic<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes,
                                                          strides, pads, group, outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost,
                                                          biasScale, outPre, outPost, outScale, depthUnroll, dilation);

/*    char software_buf[20];
    snprintf(software_buf, 20, "software_outW_%02llu.txt", inWdims[3]);

    FILE *software_outW = fopen(software_buf, "w");
    for (int i = 0; i < outWdims[0] * outWdims[1] * outWdims[2] * outWdims[3]; i++)
        fprintf(software_outW, "%d,", outW[i]);
    fprintf(software_outW, "\n");
    fclose(software_outW);*/

    /*    libjit_quantized_convolution_generic<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes,
       strides, pads, group, outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale, depthUnroll,
       dilation);*/

    /*    dlha_conv<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes, strides,
                                                             pads, group, outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost, biasScale,
                                                             outPre, outPost, outScale, depthUnroll, dilation);*/

/*    if (inWdims[3] == 1) {
        dlha_conv<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes, strides, pads, group, outOffset,
                                   inOffset, filterOffset, biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale, depthUnroll,
                                   dilation);

        FILE *hardware_outW = fopen("hardware_outW.txt", "w");
        for (int i = 0; i < outWdims[0] * outWdims[1] * outWdims[2] * outWdims[3]; i++)
            fprintf(hardware_outW, "%d,", outW[i]);
        fprintf(hardware_outW, "\n");
        fclose(hardware_outW);

        libjit_quantized_convolution_generic<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes,
                                                              strides, pads, group, outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost,
                                                              biasScale, outPre, outPost, outScale, depthUnroll, dilation);

        FILE *software_outW = fopen("software_outW.txt", "w");
        for (int i = 0; i < outWdims[0] * outWdims[1] * outWdims[2] * outWdims[3]; i++)
            fprintf(software_outW, "%d,", outW[i]);
        fprintf(software_outW, "\n");
        fclose(software_outW);
    } else if (inWdims[3] == 32) {
        *//*        dlha_conv<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes, strides, pads, group,
           outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost, biasScale, outPre, outPost, outScale, depthUnroll, dilation); FILE
           *our_software = fopen("our_software.txt", "w"); for (int i = 0; i < outWdims[1] * outWdims[2] * outWdims[3]; i++) fprintf(our_software,
           "%d,", outW[i]); fprintf(our_software, "\n"); fclose(our_software);*//*

        libjit_quantized_convolution_generic<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes,
                                                              strides, pads, group, outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost,
                                                              biasScale, outPre, outPost, outScale, depthUnroll, dilation);

        *//*        FILE *their_software = fopen("their_software.txt", "w");
                for (int i = 0; i < outWdims[1] * outWdims[2] * outWdims[3]; i++) fprintf(their_software, "%d,", outW[i]);
                fprintf(their_software, "\n");
                fclose(their_software);*//*
    } else {
        libjit_quantized_convolution_generic<int8_t, int32_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes,
                                                              strides, pads, group, outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost,
                                                              biasScale, outPre, outPost, outScale, depthUnroll, dilation);
    }*/
}

/*void libjit_convolution_i8_i8(int8_t *outW, const int8_t *inW, const int8_t *filterW, const int8_t *biasW, const dim_t *outWdims,
                              const dim_t *inWdims, const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernelSizes, const dim_t *strides,
                              const dim_t *pads, dim_t group, int32_t outOffset, int32_t inOffset, int32_t filterOffset, int32_t biasOffset,
                              int32_t biasPre, int32_t biasPost, int32_t biasScale, int32_t outPre, int32_t outPost, int32_t outScale,
                              unsigned depthUnroll, dim_t dilation) {
    libjit_quantized_convolution_generic<int8_t, int8_t>(outW, inW, filterW, biasW, outWdims, inWdims, filterWdims, biasWdims, kernelSizes, strides,
                                                         pads, group, outOffset, inOffset, filterOffset, biasOffset, biasPre, biasPost, biasScale,
                                                         outPre, outPost, outScale, depthUnroll, dilation);
}*/

void libjit_conv_transpose_f(float *outW, const float *inW, const float *filterW, const float *biasW, const dim_t *outWdims, const dim_t *inWdims,
                             const dim_t *filterWdims, const dim_t *biasWdims, const dim_t *kernels, const dim_t *strides, const dim_t *pads,
                             dim_t group, dim_t dilation) {
    // NHWC format is assumed
    dim_t p = sizeof(float);
    memset(outW, 0, outWdims[0] * outWdims[1] * outWdims[2] * outWdims[3] * p);

    dim_t pad_t = pads[0];
    dim_t pad_l = pads[1];
    dim_t stride_h = strides[0];
    dim_t stride_w = strides[1];
    dim_t kernel_h = kernels[0];
    dim_t kernel_w = kernels[1];
    dim_t outCperG = outWdims[3] / group;
    dim_t inCperG = inWdims[3] / group;

    // For each input in the batch:
    for (dim_t n = 0; n < inWdims[0]; n++) {

        // Initialize the outputs with the bias.
        libjit_conv_init_output_with_bias(n, outW, biasW, outWdims, biasWdims);

        // For each group of input channels:
        for (dim_t g = 0; g < group; g++) {
            for (dim_t d = g * inCperG; d < (g + 1) * inCperG; d++) {
                ssize_t x = -(ssize_t)pad_t;
                for (dim_t bx = 0; bx < inWdims[1]; bx++, x += stride_h) {
                    ssize_t y = -(ssize_t)pad_l;
                    for (dim_t by = 0; by < inWdims[2]; by++, y += stride_w) {
                        float grad = inW[libjit_getXYZW(inWdims, n, bx, by, d)];

                        for (dim_t kx = 0; kx < kernel_h; kx++) {
                            for (dim_t ky = 0; ky < kernel_w; ky++) {
                                ssize_t ax = x + kx * dilation;
                                ssize_t ay = y + ky * dilation;

                                if (ax < 0 || ay < 0 || ax >= (ssize_t)outWdims[1] || ay >= (ssize_t)outWdims[2]) {
                                    continue;
                                }

                                for (dim_t c = 0; c < outCperG; c++) {
                                    dim_t outIndex = libjit_getXYZW(outWdims, n, (dim_t)ax, (dim_t)ay, (g * outCperG + c));
                                    dim_t inIndex = libjit_getXYZW(filterWdims, c, kx, ky, d);
                                    outW[outIndex] += filterW[inIndex] * grad;
                                }
                            }
                        }
                    } // W
                }     // H
            }         // C
        }             // G
    }                 // N
}

void libjit_convolution_grad_f(float *inG, const float *outG, const float *inW, float *filterG, float *biasG, const float *filterW,
                               const dim_t *outGdims, const dim_t *inWdims, const dim_t *filterGdims, const dim_t *kernels, const dim_t *strides,
                               const dim_t *pads, dim_t group, dim_t dilation) {
    // NHWC format is assumed
    // Clear inG, filterG, and biasG
    dim_t p = sizeof(float);
    memset(inG, 0, inWdims[0] * inWdims[1] * inWdims[2] * inWdims[3] * p);
    memset(filterG, 0, filterGdims[0] * filterGdims[1] * filterGdims[2] * filterGdims[3] * p);
    memset(biasG, 0, outGdims[3] * p);

    dim_t pad_t = pads[0];
    dim_t pad_l = pads[1];
    dim_t stride_h = strides[0];
    dim_t stride_w = strides[1];
    dim_t kernel_h = kernels[0];
    dim_t kernel_w = kernels[1];
    dim_t inCperG = inWdims[3] / group;
    dim_t outCperG = outGdims[3] / group;

    // For each input in the batch:
    for (dim_t n = 0; n < outGdims[0]; n++) {
        // For each group of input channels:
        for (dim_t g = 0; g < group; g++) {
            for (dim_t d = g * outCperG; d < (g + 1) * outCperG; d++) {
                ssize_t x = -(ssize_t)pad_t;
                for (dim_t bx = 0; bx < outGdims[1]; bx++, x += stride_h) {
                    ssize_t y = -(ssize_t)pad_l;
                    for (dim_t by = 0; by < outGdims[2]; by++, y += stride_w) {
                        float grad = outG[libjit_getXYZW(outGdims, n, bx, by, d)];

                        for (dim_t kx = 0; kx < kernel_h; kx++) {
                            for (dim_t ky = 0; ky < kernel_w; ky++) {
                                ssize_t ax = x + kx * dilation;
                                ssize_t ay = y + ky * dilation;

                                if (ax < 0 || ay < 0 || ax >= (ssize_t)inWdims[1] || ay >= (ssize_t)inWdims[2]) {
                                    continue;
                                }

                                for (dim_t c = 0; c < inCperG; c++) {
                                    inG[libjit_getXYZW(inWdims, n, (dim_t)ax, (dim_t)ay, g * inCperG + c)] +=
                                        filterW[libjit_getXYZW(filterGdims, d, kx, ky, c)] * grad;
                                    filterG[libjit_getXYZW(filterGdims, d, kx, ky, c)] +=
                                        inW[libjit_getXYZW(inWdims, n, (dim_t)ax, (dim_t)ay, g * inCperG + c)] * grad;
                                }
                            }
                        }

                        biasG[d] += grad;
                    } // W
                }     // H
            }         // C
        }             // G
    }                 // N
}
}
