#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

void error(const char* str, ...) {
    va_list args;
    va_start(args, str);
    vfprintf(stderr, str, args);
    exit(-1);
}

void dump_to_file(const char* filename, const void* data, size_t size) {
    FILE* dumpfile;
    if (fopen_s(&dumpfile, filename, "wb")) {
        error(false, "Failed to open dump file.");
    }
    if (fwrite(data, size, 1, dumpfile) < 1) error("Failed to dump file %s.", filename);
    fclose(dumpfile);
}
