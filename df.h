#ifndef DF_H
#define DF_H

#include "view.h"

#include <stddef.h>

float parabola_intersect(struct view_f f, size_t p, size_t q);

void dist_transform_1d(struct view_f f, struct view_st v, struct view_f z);

void dist_transform_2d(float* img, size_t w, size_t h);

void dist_transform_axis(float* img, float* z_2d, size_t* v_2d, size_t w, size_t h);

void transpose_cpy(float* dest, float* src, size_t w, size_t h);
void transpose_cpy_sqrt(float* dest, float* src, size_t w, size_t h);

#endif
