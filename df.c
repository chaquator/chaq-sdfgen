#include "df.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

// intersection of 2 parabolas, not defined if both parabolas have vertex y's at infinity
float parabola_intersect(struct view_f f, size_t p, size_t q) {
    assert(p != q);
    assert((ptrdiff_t)p < (f.end - f.start));
    assert((ptrdiff_t)q < (f.end - f.start));
    assert(!isinf(f.start[p]));
    assert(!isinf(f.start[q]));

    float fp = (float)p;
    float fq = (float)q;
    return ((f.start[q] - f.start[p]) + ((fq * fq) - (fp * fp))) / (2 * (fq - fp));
}

// Compute euclidean distance transform in 1d using passed in buffers
// Reference: Distance Transforms of Sampled Functions (P. Felzenszwalb, D. Huttenlocher):
//      http://cs.brown.edu/people/pfelzens/dt/
// f -- single row buffer of parabola heights, sized N
// v -- vertices buffer, sized N
// z -- break point buffer, associates z[n] with v[n]'s right bound, sized N-1
void dist_transform_1d(struct view_f f, struct view_st v, struct view_f z) {
    assert((f.end - f.start) > 0);
    assert((v.end - v.start) > 0);
    assert((z.end - z.start) > 0);
    assert((v.end - v.start) == (f.end - f.start));
    assert((z.end - z.start) == (f.end - f.start) - 1);

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
    }

    // Part 2: Populate f using lower envelope
    size_t j = 0;
    for (ptrdiff_t q = 0; q < (f.end - f.start); ++q) {
        // Seek break point past q
        while (j < k && z.start[j] < (float)q) ++j;
        // Set point at f to parabola (originating at v[j])
        size_t v_k = v.start[j];
        float displacement = (float)q - (float)v_k;
        f.start[q] = displacement * displacement + f.start[v_k];
    }
}

void dist_transform_2d(float* img, size_t w, size_t h) {
    // allocate auxiliary memory
    size_t dim = w > h ? w : h;
    size_t other_dim = w > h ? h : w;
    // (high_dim-1)*(low_dim) is sufficient memory for both orientations
    // given that the requirement for z per row is N-1
    float* z_2d = malloc((dim - 1) * other_dim * sizeof(float));
    size_t* v_2d = malloc(dim * dim * sizeof(size_t));

    // compute 1d for all rows
    for (size_t y = 0; y < h; ++y) {
        // partition img, z, and v into views and pass into dist transform
    }

    // transpose
    // ow my cache locality

    // compute 1d for all rows (now for all columns)
    for (size_t x = 0; x < w; ++x) {
        // do the thing
    }

    // tranpose
    // yeah one more why not
}
