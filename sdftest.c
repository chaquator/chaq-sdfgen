#include <math.h>
#include <stdio.h>

#include "df.h"
#include "view.h"

// TODO: write testing function for various tests

int main() {
    float f[] = {0, 1, 2, 3, 4};
    size_t v[sizeof(f) / sizeof(float)] = {0};
    float z[sizeof(f) / sizeof(float) + 1] = {0};
    dist_transform_1d((struct view_f){f, f + sizeof(f) / sizeof(float)},
                      (struct view_st){v, v + sizeof(v) / sizeof(size_t)},
                      (struct view_f){z, z + sizeof(z) / sizeof(float)});
    return 0;
}
