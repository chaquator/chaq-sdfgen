#include <stdio.h>

#include "df.h"
#include "view.h"

int main() {
    float f[] = {7.1f, 5.9f, 4.f, 2.4f, 0.f};
    size_t v[sizeof(f) / sizeof(float)] = {};
    float z[sizeof(f) / sizeof(float) + 1] = {};
    dist_transform_1d((struct view_f){f, f + sizeof(f) / sizeof(float)},
                      (struct view_st){v, v + sizeof(v) / sizeof(size_t)},
                      (struct view_f){z, z + sizeof(z) / sizeof(float)});
    return 0;
}
