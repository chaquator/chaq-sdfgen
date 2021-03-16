#include "df.h"

#include <assert.h>
#include <math.h>

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
// z -- intersections buffer, sized N+1
void dist_transform_1d(struct view_f f, struct view_st v, struct view_f z) {
    assert((f.end - f.start) > 0);
    assert((v.end - v.start) > 0);
    assert((z.end - z.start) > 0);
    assert((v.end - v.start) == (f.end - f.start));
    assert((z.end - z.start) == (f.end - f.start) + 1);

    // Single-cell is already complete
    if ((f.end - f.start) == 1) return;

    // Part 1: Compute lower envelope as a set of break points and vertices
    // Start at the first non-infinity parabola
    ptrdiff_t offset = 0;
    while (isinf(f.start[offset]) && offset < (f.end - f.start)) ++offset;

    // If lower envelope is all at infinity, we have an empty row, this is complete as far as we care
    if (offset == (f.end - f.start)) return;

    v.start[0] = offset;
    z.start[0] = -INFINITY;
    z.start[1] = INFINITY;

    ptrdiff_t k = 0;
    for (ptrdiff_t q = offset + 1; q < (f.end - f.start); ++q) {
        if (isinf(f.start[q])) continue;
        // Calculate intersection of current parabola and next candidate
        float s = parabola_intersect(f, v.start[k], q);

        // If this intersection comes before current left-bound, we must back up and change the necessary break point
        while (s <= z.start[k]) {
            --k;
            assert(k >= 0);
            s = parabola_intersect(f, v.start[k], q);
        }
        // Once we found a suitable break point, update the structure
        ++k;
        v.start[k] = q;
        z.start[k] = s;
        z.start[k + 1] = INFINITY;
    }

    // Part 2: Populate f using lower envelope
    size_t j = 0;
    for (ptrdiff_t q = 0; q < (f.end - f.start); ++q) {
        // Seek break-point past q
        while (z.start[j + 1] < q) ++j;
        // Set point at f to parabola (originating at v[j])
        size_t v_k = v.start[j];
        f.start[q] = (q - v_k) * (q - v_k) + f.start[v_k];
    }
}

// TODO: optimize out the N+1 to N
// Not only does the highest right bound being set to infinity not matter if you keep track of the # of hulls
// But I don't think the lowest left bound being -infinity is necessary either if you move some stuff out
