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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include <algorithm>
#include <array>

#include <assert.h>
#include <inttypes.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <vector>

#include "dlha.h"

/// This is an example demonstrating how to use auto-generated bundles and
/// create standalone executables that can perform neural network computations.
/// This example loads and runs the compiled resnet50 network model.
/// This example is using the dynamic bundle API (default).

#define DEFAULT_HEIGHT 224
#define DEFAULT_WIDTH 224

//===----------------------------------------------------------------------===//
//                   Image processing helpers
//===----------------------------------------------------------------------===//
std::vector<std::string> inputImageFilenames;

/// \returns the index of the element at x,y,z,w.
size_t getXYZW(const size_t *dims, size_t x, size_t y, size_t z, size_t w) {
    return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) + (z * dims[3]) + w;
}

/// \returns the index of the element at x,y,z.
size_t getXYZ(const size_t *dims, size_t x, size_t y, size_t z) { return (x * dims[1] * dims[2]) + (y * dims[2]) + z; }

/// Reads a PNG image from a file into a newly allocated memory block \p imageT
/// representing a WxHxNxC tensor and returns it. The client is responsible for
/// freeing the memory block.
bool readPngImage(const char *filename, uint8_t *&imageT, size_t *imageDims) {
    unsigned char header[8];
    // open file and test for it being a png.
    FILE *fp = fopen(filename, "rb");
    // Can't open the file.
    if (!fp) {
        return true;
    }

    // Validate signature.
    size_t fread_ret = fread(header, 1, 8, fp);
    if (fread_ret != 8) {
        return true;
    }
    if (png_sig_cmp(header, 0, 8)) {
        return true;
    }

    // Initialize stuff.
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        return true;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        return true;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        return true;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);

    size_t width = png_get_image_width(png_ptr, info_ptr);
    size_t height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    const bool isGray = color_type == PNG_COLOR_TYPE_GRAY;
    const size_t numChannels = isGray ? 1 : 3;
    assert(isGray == 1 && "Invalid image, not gray");

    (void)bit_depth;
    assert(bit_depth == 8 && "Invalid image");
    assert((color_type == PNG_COLOR_TYPE_RGB_ALPHA || color_type == PNG_COLOR_TYPE_RGB || isGray) && "Invalid image");
    bool hasAlpha = (color_type == PNG_COLOR_TYPE_RGB_ALPHA);

    int number_of_passes = png_set_interlace_handling(png_ptr);
    (void)number_of_passes;
    assert(number_of_passes == 1 && "Invalid image");

    png_read_update_info(png_ptr, info_ptr);

    // Error during image read.
    if (setjmp(png_jmpbuf(png_ptr))) {
        return true;
    }

    auto *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
    for (size_t y = 0; y < height; y++) {
        row_pointers[y] = (png_byte *) malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, info_ptr);

    imageDims[0] = width;
    imageDims[1] = height;
    imageDims[2] = numChannels;
    imageT = static_cast<uint8_t *>(calloc(1, width * height * numChannels * sizeof(uint8_t)));

    for (size_t row_n = 0; row_n < height; row_n++) {
        png_byte *row = row_pointers[row_n];
        for (size_t col_n = 0; col_n < width; col_n++) {
            png_byte *ptr = &(row[col_n * (hasAlpha ? (numChannels + 1) : numChannels)]);
            imageT[getXYZ(imageDims, row_n, col_n, 0)] = int(ptr[0]);
        }
    }

    for (size_t y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) NULL);
    fclose(fp);
    printf("Loaded image: %s\n", filename);

    return false;
}

/// Loads and normalizes all PNGs into a tensor memory block \p resultT in the
/// NCHW 3x224x224 format.
static void loadImagesAndPreprocess(const std::vector<std::string> &filenames, uint8_t *&resultT, size_t *resultDims) {
    assert(filenames.size() > 0 && "There must be at least one filename in filenames");
//    std::pair<float, float> range = std::make_pair(0., 1.0);
    unsigned numImages = filenames.size();
    // N x C x H x W
    resultDims[0] = numImages;
    resultDims[1] = 1;
    resultDims[2] = DEFAULT_HEIGHT;
    resultDims[3] = DEFAULT_WIDTH;
    size_t resultSizeInBytes = numImages * resultDims[1] * DEFAULT_HEIGHT * DEFAULT_WIDTH * sizeof(uint8_t);
    resultT = static_cast<uint8_t *>(malloc(resultSizeInBytes));

    size_t dims[3];
    bool loadSuccess = !readPngImage(filenames[0].c_str(), resultT, dims);
    assert(loadSuccess && "Error reading input image.");
    (void)loadSuccess;

    assert((dims[0] == DEFAULT_HEIGHT && dims[1] == DEFAULT_WIDTH) && "All images must have the same Height and Width");

/*    for (int i = 0; i < resultDims[2]; i++) {
      for (int j = 0; j < resultDims[3]; j++) {
        if (j < resultDims[3] - 1)
          printf("%u ", resultT[i * resultDims[3] + j]);
        else
          printf("%u", resultT[i * resultDims[3] + j]);
      }
      printf("\n");
    }
    printf("\n");*/
    printf("Loaded images size in bytes is: %u\n", resultSizeInBytes);
}

/// Parse images file names into a vector.
void parseCommandLineOptions(int argc, char **argv) {
    int arg = 1;
    while (arg < argc) {
        inputImageFilenames.push_back(argv[arg++]);
    }
}

//===----------------------------------------------------------------------===//
//                 Wrapper code for executing a bundle
//===----------------------------------------------------------------------===//
/// Find in the bundle's symbol table a weight variable whose name starts with
/// \p name.
const SymbolTableEntry *getWeightVar(const BundleConfig &config, const char *name) {
    for (unsigned i = 0, e = config.numSymbols; i < e; ++i) {
        if (!strncmp(config.symbolTable[i].name, name, strlen(name))) {
            return &config.symbolTable[i];
        }
    }
    return nullptr;
}

/// Find in the bundle's symbol table a mutable weight variable whose name
/// starts with \p name.
const SymbolTableEntry &getMutableWeightVar(const BundleConfig &config, const char *name) {
    const SymbolTableEntry *mutableWeightVar = getWeightVar(config, name);
    if (!mutableWeightVar) {
        printf("Expected to find variable '%s'\n", name);
    }
    assert(mutableWeightVar && "Expected to find a mutable weight variable");
    assert(mutableWeightVar->kind != 0 && "Weight variable is expected to be mutable");
    return *mutableWeightVar;
}

/// Allocate an aligned block of memory.
void *alignedAlloc(const BundleConfig &config, size_t size) {
    void *ptr;
    // Properly align the memory region.
    int res = posix_memalign(&ptr, config.alignment, size);
    assert(res == 0 && "posix_memalign failed");
    assert((size_t)ptr % config.alignment == 0 && "Wrong alignment");
    memset(ptr, 0, size);
    (void)res;
    return ptr;
}

/// Initialize the constant weights memory block by loading the weights from the
/// weights file.
static uint8_t *initConstantWeights(const char *weightsFileName, const BundleConfig &config) {
    // Load weights.
    FILE *weightsFile = fopen(weightsFileName, "rb");
    if (!weightsFile) {
        fprintf(stderr, "Could not open the weights file: %s\n", weightsFileName);
        exit(1);
    }
    fseek(weightsFile, 0, SEEK_END);
    size_t fileSize = ftell(weightsFile);
    fseek(weightsFile, 0, SEEK_SET);
    uint8_t *baseConstantWeightVarsAddr = static_cast<uint8_t *>(alignedAlloc(config, fileSize));
    printf("Allocated weights of size: %u\n", fileSize);
    printf("Expected weights of size: %" PRIu64 "\n", config.constantWeightVarsMemSize);
    assert(fileSize == config.constantWeightVarsMemSize && "Wrong weights file size");
    int result = fread(baseConstantWeightVarsAddr, fileSize, 1, weightsFile);
    if (result != 1) {
        perror("Could not read the weights file");
    } else {
        printf("Loaded weights of size: %u from the file %s\n", fileSize, weightsFileName);
    }
    fclose(weightsFile);
    return baseConstantWeightVarsAddr;
}

/// The assumed layout of the area for mutable WeightVars is:
/// data | gpu_0/data | results
static uint8_t *allocateMutableWeightVars(const BundleConfig &config) {
    auto *weights = static_cast<uint8_t *>(alignedAlloc(config, config.mutableWeightVarsMemSize));
    printf("Allocated mutable weight variables of size: %" PRIu64 "\n", config.mutableWeightVarsMemSize);
    return weights;
}

/// Dump the result of the inference by looking at the results vector and
/// finding the index of the max element.
static void dumpInferenceResults(const BundleConfig &config, uint8_t *mutableWeightVars) {
    const SymbolTableEntry &outputWeights = getMutableWeightVar(config, "output");

    int maxIdx[5] = {0};
    float maxValue[5] = {0.0};
    float *results = (float *)(mutableWeightVars + outputWeights.offset);

/*    for (int j = 0; j < 5; j++){
      for (int i = 0; i < (int) outputWeights.size; ++i) {
          if(results[i] > maxValue[j]){
            if (j > 0){
              for (int k = 0; k<5; k++){
                if (results[i] != maxValue[k]){
                  maxValue[j] = results[i];
                  maxIdx[j] = i;
                }
              }
            }
            else{
              maxValue[j] = results[i];
              maxIdx[j] = i;
            }
          }
        }
      }
    printf("Results: [");
    for (int j = 0; j < 5; j++) printf("%u,", maxIdx[j]);
    printf("]\n");

    printf("Confidence: [");
    for (int j = 0; j < 5; j++) printf("%f,", maxValue[j]);
    printf("]\n");*/

//    printf("\n");

    int _maxIdx = 0;
    float _maxValue = 0;

    for (int i = 0; i < (int) outputWeights.size; ++i) {
        if (results[i] > _maxValue) {
            _maxValue = results[i];
            _maxIdx = i;
        }
    }
    printf("Result: %u\n", _maxIdx);
    printf("Confidence: %f\n", _maxValue);
}

/// The assumed layout of the area for mutable WeightVars is:
/// data | gpu_0/data | results
static uint8_t *initMutableWeightVars(const BundleConfig &config) {
    uint8_t *mutableWeightVarsAddr = allocateMutableWeightVars(config);
    size_t inputDims[4];
    uint8_t *inputT{nullptr};
    loadImagesAndPreprocess(inputImageFilenames, inputT, inputDims);

    // Copy image data into the gpu_0/data input variable in the mutableWeightVars area.
    size_t imageDataSizeInBytes = inputDims[0] * inputDims[1] * inputDims[2] * inputDims[3] * sizeof(uint8_t);
    printf("Copying image data into mutable weight vars: %u bytes\n", imageDataSizeInBytes);

    const SymbolTableEntry &inputGPUDataVar = getMutableWeightVar(config, "input");
    memcpy(mutableWeightVarsAddr + inputGPUDataVar.offset, inputT, imageDataSizeInBytes);

    return mutableWeightVarsAddr;
}

static uint8_t *initActivations(const BundleConfig &config) { return static_cast<uint8_t *>(alignedAlloc(config, config.activationsMemSize)); }


int main(int argc, char **argv) {
#ifdef DEBUG
    printf("HI JOST IS IN DEBUG");
#endif

    parseCommandLineOptions(argc, argv);
    // Allocate and initialize constant and mutable weights.
    uint8_t *constantWeightVarsAddr = initConstantWeights("dlha.weights.bin", dlha_config);
    uint8_t *mutableWeightVarsAddr = initMutableWeightVars(dlha_config);
    uint8_t *activationsAddr = initActivations(dlha_config);

    // Perform the computation.
    dlha(constantWeightVarsAddr, mutableWeightVarsAddr, activationsAddr);

    // Report the results.
    dumpInferenceResults(dlha_config, mutableWeightVarsAddr);

    // Free all resources.
    free(activationsAddr);
    free(constantWeightVarsAddr);
    free(mutableWeightVarsAddr);

    return 0;
}

