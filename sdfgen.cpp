#include <cstdint>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <string_view>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std::literals;

void usage() {
    auto usage = "usage: chaq_sdf -i file -o file [-s n]\n"
                 "    -i file: input file\n"
                 "    -o file: output file\n"
                 "    -s n: spread radius in pixels (default 4)"sv;
    std::cout << usage << '\n';
}

[[noreturn]] void error(std::string_view str, bool print_usage = false) {
    if (print_usage) usage();
    std::cerr << "Error: " << str << '\n';
    exit(-1);
}

void dump_file(const char* filename, const unsigned char* data, int width, int height, int channels) {
    std::ofstream dump(filename, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
    dump.write((char*)&data[0], width * height * channels);
}

int main(int argc, char** argv) {
    std::string_view infile;
    std::string_view outfile;
    std::size_t spread = 4;

    // process arguments
    for (int i = 0; i < argc; ++i) {
        if (argv[i][0] != '-') continue;

        switch (argv[i][1]) {
        case 'i': {
            ++i;
            if (i >= argc) {
                error("No input file specified.", true);
            }
            infile = std::string_view(argv[i]);
        } break;
        case 'o': {
            ++i;
            if (i >= argc) {
                error("No output file specified.", true);
            }
            outfile = std::string_view(argv[i]);
        } break;
        case 's': {
            ++i;
            if (i >= argc) {
                error("No number specified with spread.", true);
            }
            try {
                spread = static_cast<std::size_t>(std::stoull(argv[i]));
            } catch (std::exception& e) {
                error("Invalid value given for spread.", true);
            }
        } break;
        }
    }
    std::printf("Infile: %s\nOutfile: %s\nSpread: %lu\n", infile.data(), outfile.data(), spread);

    if (infile.data() == nullptr) error("No input file specified.", true);
    if (outfile.data() == nullptr) error("No output file specified.", true);
    if (spread <= 0) error("Spread must be a positive integer.", true);

    // 2 channels sufficient to get alpha data of image
    int w;
    int h;
    int n;
    int c = 2;
    unsigned char* data = stbi_load(infile.data(), &w, &h, &n, c);

    if (data == nullptr) error("Input file could not be opened.");

    std::printf("Filename: %s\nw: %d, h: %d, channels: %d\n", infile.data(), w, h, n);

    stbi_image_free(data);

    return 0;
}
