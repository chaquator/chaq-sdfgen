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

static void usage() {
    const char* usage =
        "usage: chaq_sdfgen -i file -o file [-s n] [-ahln]\n"
        "    -i file: input file\n"
        "    -o file: output file\n"
        "    -s n: spread radius in pixels (default 4)\n"
        "    -a: asymmetric spread (disregard negative distances, becomes unsinged distance transformation)\n"
        "        (default: symmetric)\n"
        "    -h: show the usage\n"
        "    -l: test pixel based on image luminance (default: tests based on alpha channel)\n"
        "    -n: invert alpha test; values below threshold will be counted as \"inside\" (default: not inverted),\n";
    puts(usage);
}

// transforms input image data into boolean buffer
static void transform_img_to_bool(const unsigned char* img_in, bool* bool_out, size_t width, size_t height,
                                  size_t stride, size_t offset, bool test_above) {
    for (size_t i = 0; i < width * height; ++i) {
        unsigned char threshold = 127;
        bool pixel = test_above ? img_in[i * stride + offset] > threshold : img_in[i * stride + offset] < threshold;
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
static void transform_float_to_byte(const float* float_in, unsigned char* byte_out, size_t width, size_t height,
                                    size_t spread, bool asymmetric) {
    for (size_t i = 0; i < width * height; ++i) {
        // clamped linear remap
        float s_min = asymmetric ? 0 : -(float)spread;
        float s_max = (float)spread;
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

    size_t test_channel = 1;
    bool test_above = true;
    bool asymmetric = false;
    size_t spread = 4;

    // process arguments
    for (int i = 0; i < argc; ++i) {
        if (argv[i][0] != '-') continue;

        switch (argv[i][1]) {
            // i - input file
        case 'i': {
            ++i;
            if (i >= argc) {
                usage();
                error("No input file specified.");
            }
            infile = argv[i];
        } break;
            // o - output file
        case 'o': {
            ++i;
            if (i >= argc) {
                usage();
                error("No output file specified.");
            }
            outfile = argv[i];
        } break;
            // s - spread parameter
        case 's': {
            ++i;
            if (i >= argc) {
                usage();
                error("No number specified with spread.");
            }
            spread = strtoull(argv[i], NULL, 10);
        } break;
            // flags
        default: {
            size_t j = 1;
            while (argv[i][j]) {
                switch (argv[i][j]) {
                    // h - help
                case 'h': {
                    usage();
                    return 0;
                }
                    // n - invert (test for below threshold instead of above)
                case 'n': {
                    test_above = false;
                } break;
                    // l - test based on luminance
                case 'l': {
                    test_channel = 0;
                } break;
                    // a - asymmetric spread
                case 'a': {
                    asymmetric = true;
                } break;
                }
                ++j;
            }

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

    transform_img_to_bool(img_original, img_bool, (size_t)w, (size_t)h, (size_t)c * sizeof(unsigned char), test_channel,
                          test_above);
    stbi_write_png("god.png", w, h, 1, img_bool, w * (int)sizeof(bool));

    stbi_image_free(img_original);

    // compute 2d sdf image
    float* img_float_inside = malloc((size_t)(w * h) * sizeof(float));
    if (img_float_inside == NULL) error("img_float_inside malloc failed.");

    transform_bool_to_float(img_bool, img_float_inside, (size_t)w, (size_t)h, true);

    dist_transform_2d(img_float_inside, (size_t)w, (size_t)h);

    unsigned char* img_byte = malloc((size_t)(w * h) * sizeof(unsigned char));
    if (img_byte == NULL) error("img_byte malloc failed.");
    transform_float_to_byte(img_float_inside, img_byte, (size_t)w, (size_t)h, spread, asymmetric);

    free(img_bool);
    free(img_float_inside);

    stbi_write_png(outfile, w, h, 1, img_byte, w * (int)sizeof(unsigned char));
    free(img_byte);

    return 0;
}

// TODO:
// - compute 2d sdf for "outside" pixels
// - consolidate inside vs outside into single float buffer
// - openMP the whole thing
