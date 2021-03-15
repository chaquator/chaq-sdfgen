#ifndef DF_H
#define DF_H

#include "view.h"

#include <stddef.h>

#define BIG 1e20

float parabola_intersect(struct view_f f, size_t p, size_t q);

void dist_transform_1d(struct view_f f, struct view_st v, struct view_f z);

#endif
