#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void usage() {
    const char* usage = "usage: chaq_sdf -i file -o file [-s n]\n"
                        "    -i file: input file\n"
                        "    -o file: output file\n"
                        "    -s n: spread radius in pixels (default 4)";
    puts(usage);
}

void error(const char* str, int print_usage) {
    if (print_usage) usage();
    fprintf(stderr, "Error: %s\n", str);
    exit(-1);
}

void dump_file(const char* filename, const unsigned char* data, int width, int height, int channels) {
    FILE* dumpfile;
    if (fopen_s(&dumpfile, filename, "wb")) {
        error("Failed to open dump file.", 0);
    }
    fwrite(data, width * height * channels, 1, dumpfile);
    fclose(dumpfile);
}

int main(int argc, char** argv) {
    char* infile = NULL;
    char* outfile = NULL;
    size_t spread = 4;

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
            if (!spread) error("Invalid value given for spread. Must be a positive integer.", true);
        } break;
        }
    }
    printf("Infile: %s\nOutfile: %s\nSpread: %llu\n", infile, outfile, spread);

    if (infile == NULL) error("No input file specified.", true);
    if (outfile == NULL) error("No output file specified.", true);

    // 2 channels sufficient to get alpha data of image
    int w;
    int h;
    int n;
    int c = 2;
    unsigned char* data = stbi_load(infile, &w, &h, &n, c);

    if (data == NULL) error("Input file could not be opened.", false);

    printf("Filename: %s\nw: %d, h: %d, channels: %d\n", infile, w, h, n);

    // transform image into 1-bit image
    // feed into sdf function
    // write out

    stbi_image_free(data);

    return 0;
}
