#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "df.h"
#include "view.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// not my best work
static const char* program_name;

static void usage() {
    if (program_name == NULL) program_name = "chaq_sdf";
    printf("usage: %s -i file -o file [-s n]\n"
           "    -i file: input file\n"
           "    -o file: output file\n"
           "    -s n: spread radius in pixels (default 4)\n",
           program_name);
}

static void error(bool print_usage, const char* str, ...) {
    if (print_usage) usage();
    va_list args;
    va_start(args, str);
    vfprintf(stderr, str, args);
    exit(-1);
}

static void dump_to_file(const char* filename, const void* data, size_t size) {
    FILE* dumpfile;
    if (fopen_s(&dumpfile, filename, "wb")) {
        error(false, "Failed to open dump file.");
    }
    if (fwrite(data, size, 1, dumpfile) < 1) error(false, "Failed to dump file %s.", filename);
    fclose(dumpfile);
}

static void transform_img_to_bool(const unsigned char* img_in, bool* bool_out, size_t width, size_t height,
                                  size_t stride, size_t offset) {
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < width * height; ++i) {
        bool pixel = img_in[i * stride + offset] > 127;
        bool_out[i] = pixel;
    }
}

static void transform_bool_to_float(const bool* bool_in, float* float_out, size_t width, size_t height,
                                    bool true_is_zero) {
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < width * height; ++i) {
        float_out[i] = bool_in[i] == true_is_zero ? 0.f : INFINITY;
    }
}

int main(int argc, char** argv) {
    char* infile = NULL;
    char* outfile = NULL;
    size_t spread = 4;

    program_name = argv[0];

    // process arguments
    for (int i = 0; i < argc; ++i) {
        if (argv[i][0] != '-') continue;

        switch (argv[i][1]) {
        case 'i': {
            ++i;
            if (i >= argc) {
                error(true, "No input file specified.");
            }
            infile = argv[i];
        } break;
        case 'o': {
            ++i;
            if (i >= argc) {
                error(true, "No output file specified.");
            }
            outfile = argv[i];
        } break;
        case 's': {
            ++i;
            if (i >= argc) {
                error(true, "No number specified with spread.");
            }
            spread = strtoull(argv[i], NULL, 10);
        } break;
        }
    }

    if (!spread) error(true, "Invalid value given for spread. Must be a positive integer.");
    if (infile == NULL) error(true, "No input file specified.");
    if (outfile == NULL) error(true, "No output file specified.");

    // 2 channels sufficient to get alpha data of image
    int w;
    int h;
    int n;
    int c = 2;
    unsigned char* img_original = stbi_load(infile, &w, &h, &n, c);

    if (img_original == NULL) error(false, "Input file could not be opened.");

    // transform image into bool image
    bool* img_bool = malloc((size_t)(w * h) * sizeof(bool));
    if (img_bool == NULL) error(false, "img_bool malloc failed.");

    transform_img_to_bool(img_original, img_bool, (size_t)w, (size_t)h, (size_t)c, 1);

    stbi_image_free(img_original);

    // cmpute 2d sdf image
    float* img_float_inside = malloc((size_t)(w * h) * sizeof(float));
    if (img_float_inside == NULL) error(false, "img_float_inside malloc failed.");

    transform_bool_to_float(img_bool, img_float_inside, (size_t)w, (size_t)h, true);

    /*
    float* img_float_outside = malloc((size_t)(w * h) * sizeof(float));
    if (img_float_outside == NULL) error(false, "img_float_outside malloc failed.");

    transform_bool_to_float(img_bool, img_float_outside, (size_t)w, (size_t)h, false);
    */

    free(img_bool);

    // Split into rows, feed into
    free(img_float_inside);
    // free(img_float_outside);
    // write out

    return 0;
}
