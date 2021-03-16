#ifndef DF_H
#define DF_H

#include "view.h"

#include <stddef.h>

// results to get back from test
struct test_results {
    ptrdiff_t k;
    size_t j;
};

float parabola_intersect(struct view_f f, size_t p, size_t q);

void dist_transform_1d(struct view_f f, struct view_st v, struct view_f z, struct test_results* test);

#endif
