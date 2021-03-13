#include <fstream>
#include <iostream>
#include <omp.h>
#include <string_view>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std::literals;

void usage() {
    auto usage = "usage: chaq_sdf input_file output_file"sv;
    std::cout << usage << '\n';
}

[[noreturn]] void error(std::string_view str) {
    std::cerr << "Error: " << str << '\n';
    exit(-1);
}

void dump_file(const char* filename, const unsigned char* data, int width, int height, int channels) {
    std::ofstream dump(filename, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    dump.write((char*)&data[0], width * height * channels);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        usage();
        error("Insufficient argument count.");
    }

    // 2 channels sufficient to get alpha data of image
    int w;
    int h;
    int n;
    int c = 2;
    unsigned char* data = stbi_load(argv[1], &w, &h, &n, c);

    if (data == nullptr) error("Input file could not be opened.");

    std::printf("Filename: %s\nw: %d, h: %d, channels: %d\n", argv[1], w, h, n);

    dump_file(argv[2], data, w, h, c);

    stbi_image_free(data);

    return 0;
}
