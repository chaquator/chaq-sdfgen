#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "df.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

enum FILETYPE { FT_NONE = -1, FT_PNG, FT_BMP, FT_JPG, FT_TGA };

static void error(const char* str, ...) {
    va_list args;
    va_start(args, str);
    vfprintf(stderr, str, args);
    exit(-1);
}

/*
static void dump_to_file(const char* filename, const void* data, size_t size) {
    FILE* dumpfile;
    if (fopen_s(&dumpfile, filename, "wb")) {
        error(false, "Failed to open dump file.");
    }
    if (fwrite(data, size, 1, dumpfile) < 1) error("Failed to dump file %s.", filename);
    fclose(dumpfile);
}
*/

static void usage() {
    const char* usage =
        "usage: chaq_sdfgen [-f filetype] -i file -o file [-q n] [-s n] [-ahln]\n"
        "    -f filetype: manually specify filetype among PNG, BMP, TGA, and JPG\n"
        "        (default: deduced by output filename. if not deducable, default is png)\n"
        "    -i file: input file\n"
        "    -o file: output file\n"
        "    -q n: jpg quality (default: 100, only relevant for jpeg output)\n"
        "    -s n: spread radius in pixels (default: 4)\n"
        "    -a: asymmetric spread (disregard negative distances, becomes unsinged distance transformation)\n"
        "        (default: symmetric)\n"
        "    -h: show the usage\n"
        "    -l: test pixel based on image luminance (default: tests based on alpha channel)\n"
        "    -n: invert alpha test; values below threshold will be counted as \"inside\" (default: not inverted)";
    puts(usage);
}

// transforms input image data into boolean buffer
static void transform_img_to_bool(const unsigned char* restrict img_in, bool* restrict bool_out, size_t width,
                                  size_t height, size_t stride, size_t offset, bool test_above) {
    ptrdiff_t i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < (ptrdiff_t)(width * height); ++i) {
        unsigned char threshold = 127;
        bool pixel = test_above ? img_in[(size_t)i * stride + offset] > threshold
                                : img_in[(size_t)i * stride + offset] < threshold;
        bool_out[i] = pixel;
    }
}

// transforms boolean buffer to float buffer
static void transform_bool_to_float(const bool* restrict bool_in, float* restrict float_out, size_t width,
                                    size_t height, bool true_is_zero) {
    ptrdiff_t i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < (ptrdiff_t)(width * height); ++i) {
        float_out[i] = bool_in[(size_t)i] == true_is_zero ? 0.f : INFINITY;
    }
}

// single-channel char array output of input floats
static void transform_float_to_byte(const float* restrict float_in, unsigned char* restrict byte_out, size_t width,
                                    size_t height, size_t spread, bool asymmetric) {
    ptrdiff_t i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < (ptrdiff_t)(width * height); ++i) {
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

        float remap = (((v - s_min) * nd) / sn) + d_min;
        byte_out[(size_t)i] = (unsigned char)remap;
    }
}

static void transform_float_sub(float* restrict float_dst, float* restrict float_by, size_t width, size_t height) {
    ptrdiff_t i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < (ptrdiff_t)(width * height); ++i) {
        float bias = -1.f;
        float val = float_by[(size_t)i] > 0.f ? float_by[i] + bias : float_by[(size_t)i];
        float_dst[(size_t)i] -= val;
    }
}

static enum FILETYPE read_filetype(const char* string) {
    const char* type_table[] = {"png", "bmp", "jpg", "tga"};
    size_t n_types = sizeof(type_table) / sizeof(const char*);
    for (size_t filetype = 0; filetype < n_types; ++filetype) {
        if (strncmp(string, type_table[filetype], 3) == 0) return (enum FILETYPE)filetype;
    }
    return FT_NONE;
}

int main(int argc, char** argv) {
    omp_set_nested(1);

    char* infile = NULL;
    char* outfile = NULL;

    size_t test_channel = 1;
    bool test_above = true;
    bool asymmetric = false;
    size_t spread = 4;
    size_t quality = 100;
    enum FILETYPE filetype = FT_NONE;

    // process arguments
    for (int i = 0; i < argc; ++i) {
        if (argv[i][0] != '-') continue;

        switch (argv[i][1]) {
            // i - input file
        case 'i': {
            if (++i >= argc && infile == NULL) {
                usage();
                error("No input file specified.");
            } else if (i < argc) {
                infile = argv[i];
            }
        } break;
            // o - output file
        case 'o': {
            if (++i >= argc && outfile == NULL) {
                usage();
                error("No output file specified.");
            } else if (i < argc) {
                outfile = argv[i];
            }
        } break;
            // s - spread parameter
        case 's': {
            if (++i >= argc) {
                usage();
                error("No number specified with spread.");
            }
            spread = strtoull(argv[i], NULL, 10);
        } break;
            // q -- jpeg quality
        case 'q': {
            if (++i >= argc) {
                usage();
                error("No number specified with quality.");
            }
            quality = strtoull(argv[i], NULL, 10);
        } break;
            // f -- filetype
        case 'f': {
            if (++i >= argc) {
                usage();
                error("Filetype not specified with filetype switch.");
            }
            if ((filetype = read_filetype(argv[i])) == FT_NONE) {
                usage();
                error("Invalid filetype specified.");
            }
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

    if (!quality || quality > 100) {
        usage();
        error("Invalid value given for jpeg quality. Must be between 1-100");
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

    stbi_image_free(img_original);

    // compute 2d sdf images
    // inside -- pixel distance to INSIDE
    // outside -- pixel distance to OUTSIDE
    float* img_float_inside = malloc((size_t)(w * h) * sizeof(float));
    if (img_float_inside == NULL) error("img_float_inside malloc failed.");
    float* img_float_outside = malloc((size_t)(w * h) * sizeof(float));
    if (img_float_outside == NULL) error("img_float_outside malloc failed.");

#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            transform_bool_to_float(img_bool, img_float_inside, (size_t)w, (size_t)h, true);
            dist_transform_2d(img_float_inside, (size_t)w, (size_t)h);
        }
#pragma omp section
        {
            transform_bool_to_float(img_bool, img_float_outside, (size_t)w, (size_t)h, false);
            dist_transform_2d(img_float_outside, (size_t)w, (size_t)h);
        }
    }

    free(img_bool);

    // consolidate in the form of (inside - outside) to img_float_inside
    transform_float_sub(img_float_inside, img_float_outside, (size_t)w, (size_t)h);
    free(img_float_outside);

    // transform distance values to pixel values
    unsigned char* img_byte = malloc((size_t)(w * h) * sizeof(unsigned char));
    if (img_byte == NULL) error("img_byte malloc failed.");
    transform_float_to_byte(img_float_inside, img_byte, (size_t)w, (size_t)h, spread, asymmetric);

    free(img_float_inside);

    // deduce filetype if not specified
    char* dot = strrchr(outfile, '.');
    if (dot != NULL && filetype == FT_NONE) {
        filetype = read_filetype(dot + 1);
    }

    // output image
    switch (filetype) {
    case FT_BMP: {
        // bmp
        stbi_write_bmp(outfile, w, h, 1, img_byte);
    } break;
    case FT_JPG: {
        // jpg
        stbi_write_jpg(outfile, w, h, 1, img_byte, (int)quality);
    } break;
    case FT_TGA: {
        // tga
        stbi_write_tga(outfile, w, h, 1, img_byte);
    } break;
    case FT_PNG:
    case FT_NONE: {
        // png
        stbi_write_png(outfile, w, h, 1, img_byte, w * (int)sizeof(unsigned char));
    } break;
    }

    free(img_byte);

    return 0;
}
