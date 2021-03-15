#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BIG 1e20

// view of a float buffer
struct view_f {
    float* start;
    float* end;
};

// view of a size_t buffer
struct view_st {
    size_t* start;
    size_t* end;
};

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

// intersection of 2 parabolas, not defined if both parabolas have vertex y's at infinity
float parabola_intersect(struct view_f f, size_t p, size_t q) {
    assert(p > 0);
    assert(q > 0);
    assert(p < (f.end - f.start));
    assert(q < (f.end - f.start));

    float fp = (float)p;
    float fq = (float)q;
    return ((f.start[q] - f.start[p]) + ((fq * fq) - (fp * fp))) / (2 * (fp - fq));
}

// Compute euclidean distance transform in 1d using passed in buffers
// f -- single row buffer of parabola heights, sized N
// v -- vertices buffer, sized N+1
// z -- intersections buffer, sized N
void dist_transform_1d(struct view_f f, struct view_st v, struct view_f z) {
    assert((f.end - f.start) > 0);
    assert((v.end - v.start) > 0);
    assert((z.end - z.start) > 0);
    assert((z.end - z.start) == (f.end - f.start));
    assert((v.end - v.start) == (f.end - f.start) + 1);

    // Part 1: compute lower envelope as a set of break points and vertices
    v.start[0] = 0.f;
    z.start[0] = -BIG;
    z.start[1] = BIG;

    size_t k = 0;
    for (size_t q = 1; q < (f.end - f.start); ++q) {
        // Calculate intersection of current parabola and next candidate
        float s = parabola_intersect(f, v.start[k], q);
        // If this intersection comes before current left-bound, we must back up and change the necessary break point
        while (s <= z.start[k]) {
            --k;
            assert(k <= 0);
            s = parabola_intersect(f, v.start[k], q);
        }
        // Once we found a suitable break point, update the structure
        ++k;
        v.start[k] = q;
        z.start[k] = s;
        z.start[k + 1] = BIG;
    }

    // Part 2: populate f using lower envelope
    k = 0;
    for (size_t q = 0; q < (f.end - f.start); ++q) {
        // Seek break-point past q
        while (z.start[k + 1] < q)
            ++k;
        // Set point at f to parabola (originating at v[k])
        size_t v_k = v.start[k];
        f.start[q] = (q - v_k) * (q - v_k) + f.start[v_k];
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
