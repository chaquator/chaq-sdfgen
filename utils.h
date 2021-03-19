#include <stdbool.h>

#ifndef UTILS_H
#define UTILS_H

void error(const char* str, ...);

void dump_to_file(const char* filename, const void* data, size_t size);

#endif
