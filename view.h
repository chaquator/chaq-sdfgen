#ifndef VIEW_H
#define VIEW_H

#include <stddef.h>

// start is inculsve lower-bound
// end is exclusive upper-bound

// view of a float buffer
struct view_f {
    float* start;
    float* end;
};

// view of a size_t buffer
struct view_st {
    size_t* start;
    size_t* end;
};

#endif
