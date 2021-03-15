#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "df.h"
#include "view.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// not my best work
const char* program_name;

void usage() {
    const char* usage = "usage: %s -i file -o file [-s n]\n"
                        "    -i file: input file\n"
                        "    -o file: output file\n"
                        "    -s n: spread radius in pixels (default 4)\n";
    if (program_name == NULL) program_name = "chaq_sdf";
    printf(usage, program_name);
}

void error(const char* str, bool print_usage) {
    if (print_usage) usage();
    fprintf(stderr, "Error: %s\n", str);
    exit(-1);
}

void dump_file(const char* filename, const unsigned char* data, size_t width, size_t height, size_t channels) {
    FILE* dumpfile;
    if (fopen_s(&dumpfile, filename, "wb")) {
        error("Failed to open dump file.", 0);
    }
    fwrite(data, width * height * channels, 1, dumpfile);
    fclose(dumpfile);
}

void transform_img_to_onebit(const unsigned char* img_in, unsigned char* img_out, size_t width, size_t height,
                             size_t stride, size_t offset) {
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < width * height; ++i) {
        bool pixel = img_in[i * stride + offset] > 127;
        img_out[i] = (unsigned char)pixel;
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
                error("No input file specified.", true);
            }
            infile = argv[i];
        } break;
        case 'o': {
            ++i;
            if (i >= argc) {
                error("No output file specified.", true);
            }
            outfile = argv[i];
        } break;
        case 's': {
            ++i;
            if (i >= argc) {
                error("No number specified with spread.", true);
            }
            spread = strtoull(argv[i], NULL, 10);
        } break;
        }
    }

    if (!spread) error("Invalid value given for spread. Must be a positive integer.", true);
    if (infile == NULL) error("No input file specified.", true);
    if (outfile == NULL) error("No output file specified.", true);

    // 2 channels sufficient to get alpha data of image
    int w;
    int h;
    int n;
    int c = 2;
    unsigned char* img_original = stbi_load(infile, &w, &h, &n, c);

    if (img_original == NULL) error("Input file could not be opened.", false);

    // transform image into 1-bit image
    unsigned char* img_onebit = malloc(w * h * sizeof(unsigned char));

    if (img_onebit == NULL) error("img_onebit malloc failed.", false);

    transform_img_to_onebit(img_original, img_onebit, w, h, c, 1);

    stbi_image_free(img_original);

    // feed into sdf function
    free(img_onebit);
    // write out

    return 0;
}
