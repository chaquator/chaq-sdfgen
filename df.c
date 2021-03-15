#include "df.h"

#include <assert.h>
#include <math.h>

#define BIG INFINITY

// intersection of 2 parabolas, not defined if both parabolas have vertex y's at infinity
float parabola_intersect(struct view_f f, size_t p, size_t q) {
    assert(p >= 0);
    assert(q > 0);
    assert(p < (f.end - f.start));
    assert(q < (f.end - f.start));

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

    // Part 1: compute lower envelope as a set of break points and vertices
    v.start[0] = 0.f;
    z.start[0] = -BIG;
    z.start[1] = BIG;
    ptrdiff_t k = 0;
    for (size_t q = 1; q < (f.end - f.start); ++q) {
        // Calculate intersection of current parabola and next candidate
        float s = parabola_intersect(f, v.start[k], q);

        // If this intersection comes before current left-bound, we must back up and change the necessary break point
        while (s <= z.start[k]) {
            --k;
            assert(k > 0);
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

// TODO: test for parabolas at infinity (1(q) indicator)
