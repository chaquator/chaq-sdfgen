#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "df.h"
#include "view.h"

// TODO: write testing function for various tests
static int test_5(const char* name, float in[5], float expected[5], bool expect_match) {
    size_t v[5] = {0};
    float z[6] = {0};
    float out[5] = {0};
    struct test_results results = {.k = 0, .j = 0};
    assert(!memcpy_s(out, 5 * sizeof(float), in, 5 * sizeof(float)));
    dist_transform_1d((struct view_f){out, out + 5}, (struct view_st){v, v + 5}, (struct view_f){z, z + 6}, &results);

    int cmp = memcmp(out, expected, 5 * sizeof(float));
    bool match = cmp == 0;
    bool pass = match == expect_match;

    printf("Test: %s\n"
           "Data: { %f %f %f %f %f }\n"
           "Expected: { %f %f %f %f %f }\n"
           "Result: { %f %f %f %f %f }\n"
           "V: { %llu %llu %llu %llu %llu }\n"
           "Z: { %f %f %f %f %f %f }\n"
           "k: %lld, j: %llu\n"
           "Status: %s\n\n",
           name, (double)in[0], (double)in[1], (double)in[2], (double)in[3], (double)in[4], (double)expected[0],
           (double)expected[1], (double)expected[2], (double)expected[3], (double)expected[4], (double)out[0],
           (double)out[1], (double)out[2], (double)out[3], (double)out[4], v[0], v[1], v[2], v[3], v[4], (double)z[0],
           (double)z[1], (double)z[2], (double)z[3], (double)z[4], (double)z[5], results.k, results.j,
           pass ? "pass" : "FAILURE");

    return pass ? 1 : 0;
}

int main() {
    // behold my not best work
    if (
        // simple increasing
        test_5("increasing", (float[5]){0.f, 1.f, 2.f, 3.f, 4.f}, (float[5]){0.f, 1.f, 2.f, 3.f, 4.f}, true) +
            // random ish
            test_5("randomish", (float[5]){2.2f, 1.f, 3.6f, 3.5f, 2.7f}, (float[5]){2.f, 1.f, 2.f, 3.5f, 2.7f}, true) +
            // decreasing
            test_5("decreasing", (float[5]){4.4f, 3.3f, 2.2f, 1.1f, 0.f}, (float[5]){4.3f, 3.2f, 2.1f, 1.f, 0.f}, true)
        // dominated
        // all infinite
        // all but one infinite
        // all 0

        < 3) {
        puts("THERE WAS A FAILURE.");
    } else {
        puts("all clear");
    }
    return 0;
}