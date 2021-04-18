#include "df.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// intersection of 2 parabolas, not defined if both parabolas have vertex y's at infinity
static float parabola_intersect(struct view_f f, size_t p, size_t q) {
    float fp = (float)p;
    float fq = (float)q;
    return ((f.start[q] - f.start[p]) + ((fq * fq) - (fp * fp))) / (2 * (fq - fp));
}

// Compute euclidean distance transform in 1d using passed in buffers
// Reference: Distance Transforms of Sampled Functions (P. Felzenszwalb, D. Huttenlocher):
//      http://cs.brown.edu/people/pfelzens/dt/
// f -- single row buffer of parabola heights, sized N
// v -- vertices buffer, sized N
// h -- vertex height buffer, sized N
// z -- break point buffer, associates z[n] with v[n]'s right bound, sized N-1
static void dist_transform_1d(struct view_f f, struct view_st v, struct view_f h, struct view_f z) {
    // Single-cell is already complete
    if ((f.end - f.start) <= 1) return;

    // Part 1: Compute lower envelope as a set of break points and vertices
    // Start at the first non-infinity parabola
    size_t offset = 0;
    while (isinf(f.start[offset]) && offset < (size_t)(f.end - f.start)) ++offset;

    // If lower envelope is all at infinity, we have an empty row, this is complete as far as we care
    if (offset == (size_t)(f.end - f.start)) return;

    // First vertex is that of the first parabola
    v.start[0] = offset;

    size_t k = 0;
    for (size_t q = offset + 1; q < (size_t)(f.end - f.start); ++q) {
        // Skip parabolas at infinite heights (essentially non-existant parabolas)
        if (isinf(f.start[q])) continue;

        // Calculate intersection of current parabola and next candidate
        float s = parabola_intersect(f, v.start[k], q);

        // If this intersection comes before current left bound, we must back up and change the necessary break point
        // Skip for k == 0 because there is no left bound to look back on (it is at -infinity)
        while (k > 0 && s <= z.start[k - 1]) {
            --k;
            s = parabola_intersect(f, v.start[k], q);
        }
        // Once we found a suitable break point, update the structure
        // Right bound of current parabola is intersection
        z.start[k] = s;
        ++k;
        // Horizontal position of next parabola is vertex
        v.start[k] = q;
        // Vertical position of next parabola
        h.start[k] = f.start[q];
    }

    // Part 2: Populate f using lower envelope
    size_t j = 0;
    for (ptrdiff_t q = 0; q < (f.end - f.start); ++q) {
        // Seek break point past q
        while (j < k && z.start[j] < (float)q) ++j;
        // Set point at f to parabola (originating at v[j])
        size_t v_j = v.start[j];
        float displacement = (float)q - (float)v_j;
        f.start[q] = displacement * displacement + h.start[j];
    }
}

// compute distance transform along x-axis of image using buffers passed in
// img must be at least w*h floats large
static void dist_transform_axis(float* img, size_t w, size_t h) {
    ptrdiff_t y;
#pragma omp parallel for schedule(static)
    for (y = 0; y < (ptrdiff_t)(h); ++y) {
        // partition img, z, and v into views and pass into dist transform
        struct view_f f = {.start = img + ((size_t)y * w), .end = img + (((size_t)y + 1) * w)};
        // Verticess buffer
        size_t* v = malloc(sizeof(size_t) * (size_t)(w));
        // Vertex height buffer
        float* p = malloc(sizeof(float) * (size_t)(w));
        // Break point buffer
        float* z = malloc(sizeof(float) * (size_t)(w - 1));

        struct view_st v_v = {.start = v, .end = v + w};
        struct view_f v_p = {.start = p, .end = p + w};
        struct view_f v_z = {.start = z, .end = z + (w - 1)};

        dist_transform_1d(f, v_v, v_p, v_z);

        free(z);
        free(p);
        free(v);
    }
}

// copy transpose of src to dest, creates a dest that is h x w, where src is w x h
// w -- width dimension of src
// h -- height dimension of src
static void transpose_cpy(float* restrict dest, float* restrict src, size_t w, size_t h) {
    ptrdiff_t i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < (ptrdiff_t)(w * h); ++i) {
        size_t x = (size_t)i % w;
        size_t y = (size_t)i / w;
        dest[h * x + y] = src[(size_t)i];
    }
}

static void transpose_cpy_sqrt(float* restrict dest, float* restrict src, size_t w, size_t h) {
    ptrdiff_t i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < (ptrdiff_t)(w * h); ++i) {
        size_t x = (size_t)i % w;
        size_t y = (size_t)i / w;
        dest[h * x + y] = sqrtf(src[(size_t)i]);
    }
}

void dist_transform_2d(float* img, size_t w, size_t h) {
    // compute 1d for all rows
    dist_transform_axis(img, w, h);

    // transpose
    float* img_tpose = malloc(w * h * sizeof(float));
    transpose_cpy(img_tpose, img, w, h);

    // compute 1d for all rows (now for all columns)
    dist_transform_axis(img_tpose, h, w);

    // tranpose back while computing square root
    transpose_cpy_sqrt(img, img_tpose, h, w);

    free(img_tpose);
}
