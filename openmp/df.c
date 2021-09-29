#include "df.h"
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

// intersection of 2 parabolas, not defined if both parabolas have vertex y's at infinity
static float parabola_intersect(float* restrict f, size_t p, size_t q) {
    float p1_x = (float)p;
    float p2_x = (float)q;
    float p1_y = f[p];
    float p2_y = f[q];
    return ((p2_y - p1_y) + ((p2_x * p2_x) - (p1_x * p1_x))) / (2 * (p2_x - p1_x));
}

// Compute euclidean distance transform in 1d using passed in buffers
// Reference: Distance Transforms of Sampled Functions (P. Felzenszwalb, D. Huttenlocher):
//      http://cs.brown.edu/people/pfelzens/dt/
// img_row -- single row buffer of parabola heights
// w -- size of img_row
// y -- current y for transpose
// v -- vertices buffer, sized w
// h -- vertex height buffer, sized w
// z -- break point buffer, associates z[n] with v[n]'s right bound, sized w-1
// img_tpose_out -- output buffer for distance transform, will be populated in transpose, must be sufficiently large
// (at minimum w * y)
// do_sqrt -- whether to compute sqrt of value after computing lower envelope
static void dist_transform_1d(float* restrict img_row, size_t w, size_t y, size_t* restrict v, float* restrict h,
                              float* restrict z, float* restrict img_tpose_out, bool do_sqrt) {
    // Single-cell is already complete
    if (w <= 1) {
        // Write back to single cell
        img_tpose_out[0] = img_row[0];
        return;
    }

    // Part 1: Compute lower envelope as a set of break points and vertices
    // Start at the first non-infinity parabola
    size_t offset = 0;
    while (isinf(img_row[offset]) && offset < w) ++offset;

    // If lower envelope is all at infinity, we have an empty row, this is complete as far as we care
    if (offset == w) {
        // Because we're transposing on writeback, we need to fill empty rows
        for (size_t i = 0; i < w; ++i) {
            size_t tpose_idx = y + w * i;
            img_tpose_out[tpose_idx] = INFINITY;
        }
        return;
    }

    // First vertex is that of the first parabola
    v[0] = offset;
    h[0] = img_row[offset];

    size_t k = 0;
    for (size_t q = offset + 1; q < w; ++q) {
        // Skip parabolas at infinite heights (essentially non-existant parabolas)
        if (isinf(img_row[q])) continue;

        // Calculate intersection of current parabola and next candidate
        float s = parabola_intersect(img_row, v[k], q);

        // If this intersection comes before current left bound, we must back up and change the necessary break point
        // Skip for k == 0 because there is no left bound to look back on (it is at -infinity)
        while (k > 0 && s <= z[k - 1]) {
            --k;
            s = parabola_intersect(img_row, v[k], q);
        }
        // Once we found a suitable break point, update the structure
        // Right bound of current parabola is intersection
        z[k] = s;
        ++k;
        // Horizontal position of next parabola is vertex
        v[k] = q;
        // Vertical position of next parabola
        h[k] = img_row[q];
    }

    // Part 2: Populate img_row using lower envelope
    size_t j = 0;
    for (size_t q = 0; q < w; ++q) {
        // Seek break point past q
        while (j < k && z[j] < (float)q) ++j;

        // Set height at current position (q) along output to lower envelope
        size_t v_j = v[j];
        float displacement = (float)q - (float)v_j;

        // Output transposed
        size_t tpose_idx = y + w * q;
        img_tpose_out[tpose_idx] = displacement * displacement + h[j];

        if (do_sqrt) img_tpose_out[tpose_idx] = sqrtf(img_tpose_out[tpose_idx]);
    }
}

// Compute distance transform along x-axis of image using buffers passed in
// img must be at least w*h floats large
// Writes back to img_out in transpose which must be h*w floats large
static void dist_transform_axis(float* restrict img, size_t w, size_t h, float* restrict img_tpose_out, bool do_sqrt) {
#pragma omp parallel
    {
        ptrdiff_t y;
        // Verticess buffer
        size_t* v = malloc(sizeof(size_t) * (size_t)(w));
        // Vertex height buffer
        float* p = malloc(sizeof(float) * (size_t)(w));
        // Break point buffer
        float* z = malloc(sizeof(float) * (size_t)(w - 1));

#pragma omp for schedule(static)
        for (y = 0; y < (ptrdiff_t)(h); ++y) {
            float* img_slice = img + ((size_t)y * w);
            dist_transform_1d(img_slice, w, (size_t)y, v, p, z, img_tpose_out, do_sqrt);
        }

        free(z);
        free(p);
        free(v);
    }
}

void dist_transform_2d(float* img, size_t w, size_t h) {
    // compute 1d for all rows
    float* img_tpose = malloc(w * h * sizeof(float));

    // compute distance transform and store transposed into img_tpose
    dist_transform_axis(img, w, h, img_tpose, false);

    // now do pass on transpose and store back into original image
    dist_transform_axis(img_tpose, h, w, img, true);

    free(img_tpose);
}
