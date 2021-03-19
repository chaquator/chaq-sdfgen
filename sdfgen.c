#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "df.h"
#include "utils.h"
#include "view.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// not my best work
static const char* program_name;

static void usage() {
    if (program_name == NULL) program_name = "chaq_sdf";
    printf("usage: %s -i file -o file [-s n] [-h]\n"
           "    -i file: input file\n"
           "    -o file: output file\n"
           "    -s n: spread radius in pixels (default 4)\n"
           "    -h: show the usage\n",
           program_name);
}

// transforms input image data into boolean buffer
static void transform_img_to_bool(const unsigned char* img_in, bool* bool_out, size_t width, size_t height,
                                  size_t stride, size_t offset) {
    for (size_t i = 0; i < width * height; ++i) {
        bool pixel = img_in[i * stride + offset] < 127;
        bool_out[i] = pixel;
    }
}

// transforms boolean buffer to float buffer
static void transform_bool_to_float(const bool* bool_in, float* float_out, size_t width, size_t height,
                                    bool true_is_zero) {
    for (size_t i = 0; i < width * height; ++i) {
        float_out[i] = bool_in[i] == true_is_zero ? 0.f : INFINITY;
    }
}

// single-channel char array output of input floats
static void transform_float_to_byte(const float* float_in, unsigned char* byte_out, size_t width, size_t height) {
    for (size_t i = 0; i < width * height; ++i) {
        // clamped linear remap
        float s_min = 0.f;
        float s_max = 100.f;
        float d_min = 0.f;
        float d_max = 255.f;

        float sn = s_max - s_min;
        float nd = d_max - d_min;

        float v = float_in[i];
        v = v > s_max ? s_max : v;
        v = v < s_min ? s_min : v;

        float remap = (v - s_min) * (nd / sn) + d_min;
        byte_out[i] = (unsigned char)remap;
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
                usage();
                error("No input file specified.");
            }
            infile = argv[i];
        } break;
        case 'o': {
            ++i;
            if (i >= argc) {
                usage();
                error("No output file specified.");
            }
            outfile = argv[i];
        } break;
        case 's': {
            ++i;
            if (i >= argc) {
                usage();
                error("No number specified with spread.");
            }
            spread = strtoull(argv[i], NULL, 10);
        } break;
        case 'h': {
            usage();
            return 0;
        } break;
        }
    }

    if (!spread) {
        usage();
        error("Invalid value given for spread. Must be a positive integer.");
    }
    if (infile == NULL) {
        usage();
        error("No input file specified.");
    }
    if (outfile == NULL) {
        usage();
        error("No output file specified.");
    }

    // 2 channels sufficient to get alpha data of image
    int w;
    int h;
    int n;
    int c = 2;
    unsigned char* img_original = stbi_load(infile, &w, &h, &n, c);

    if (img_original == NULL) error("Input file could not be opened.");

    // transform image into bool image
    bool* img_bool = malloc((size_t)(w * h) * sizeof(bool));
    if (img_bool == NULL) error("img_bool malloc failed.");

    // transform_img_to_bool(img_original, img_bool, (size_t)w, (size_t)h, (size_t)c, 1);
    transform_img_to_bool(img_original, img_bool, (size_t)w, (size_t)h, (size_t)c, 0);
    stbi_write_png("god.png", w, h, 1, img_bool, w * (int)sizeof(bool));

    stbi_image_free(img_original);

    // compute 2d sdf image
    float* img_float_inside = malloc((size_t)(w * h) * sizeof(float));
    if (img_float_inside == NULL) error("img_float_inside malloc failed.");

    transform_bool_to_float(img_bool, img_float_inside, (size_t)w, (size_t)h, true);

    dist_transform_2d(img_float_inside, (size_t)w, (size_t)h);

    unsigned char* img_byte = malloc((size_t)(w * h) * sizeof(unsigned char));
    if (img_byte == NULL) error("img_byte malloc failed.");
    transform_float_to_byte(img_float_inside, img_byte, (size_t)w, (size_t)h);

    free(img_bool);
    free(img_float_inside);

    stbi_write_png(outfile, w, h, 1, img_byte, w * (int)sizeof(unsigned char));
    free(img_byte);

    return 0;
}

// TODO:
// - implement option for generating sdf based on alpha or on luminence
// - implement option for inverting alpha test
// - compute 2d sdf for "outside" pixels
// - consolidate inside vs outside into single float buffer
// - openMP the whole thing
