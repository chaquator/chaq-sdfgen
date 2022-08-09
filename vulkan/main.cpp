#include <algorithm>
#include <cstdint>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include "filetype.h"
#include "vulkan_ctx.h"

vulkan_ctx ctx;

// image related functions
struct stbi_img {
    const unsigned char* data = nullptr;
    int width = 0;
    int height = 0;
    int bytes_per_pixel = 0;
};

static std::optional<stbi_img> open_image(std::string_view filename) {
    bool use_stdin = filename == "-";

    unsigned char* stbi_data;
    int w, h, n;

    if (use_stdin) {
        spdlog::trace("Loading image from stdin");
        stbi_data = stbi_load_from_file(stdin, &w, &h, &n, 2);
    } else {
        spdlog::trace("Loading image from filename \"{}\"", filename);
        stbi_data = stbi_load(filename.data(), &w, &h, &n, 2);
    }

    if (stbi_data == nullptr) {
        spdlog::warn("Loading image failed (stbi error: {})", stbi_failure_reason());
        return std::nullopt;
    }

    spdlog::trace("Image stats:");
    spdlog::trace("W: {}, H: {}, Channels: {}", w, h, n);

    return stbi_img{
        stbi_data,
        w,
        h,
        2,
    };
}

static void write_to_stdout(void* context, void* data, int size) {
    (void)(context);
    fwrite(data, (size_t)size, 1, stdout);
}

static bool write_image(std::string_view filename, filetype::filetype file_type, const stbi_img& img, int quality) {
    using namespace std::literals::string_view_literals;

    bool use_stdout = filename == "-";

    spdlog::trace("Filename: {}", filename);
    spdlog::trace("Writing to stdout: {}", use_stdout);
    spdlog::trace("File type: {}", filetype::to_str(file_type));
    spdlog::trace("Quality: {}", quality);

    switch (file_type) {
    case filetype::bmp: {
        if (use_stdout) {
            return 0 != stbi_write_bmp_to_func(write_to_stdout, nullptr, img.width, img.height, img.bytes_per_pixel,
                                               img.data);
        } else {
            return 0 != stbi_write_bmp(filename.data(), img.width, img.height, img.bytes_per_pixel, img.data);
        }
    } break;
    case filetype::jpeg: {
        if (use_stdout) {
            return 0 != stbi_write_jpg_to_func(write_to_stdout, nullptr, img.width, img.height, img.bytes_per_pixel,
                                               img.data, quality);
        } else {
            return 0 != stbi_write_jpg(filename.data(), img.width, img.height, img.bytes_per_pixel, img.data, quality);
        }
    } break;
    case filetype::png: {
        if (use_stdout) {
            return 0 != stbi_write_png_to_func(write_to_stdout, nullptr, img.width, img.height, img.bytes_per_pixel,
                                               img.data, img.bytes_per_pixel * img.width);
        } else {
            return 0 != stbi_write_png(filename.data(), img.width, img.height, img.bytes_per_pixel, img.data,
                                       img.bytes_per_pixel * img.width);
        }
    } break;
    case filetype::tga: {
        if (use_stdout) {
            return 0 != stbi_write_tga_to_func(write_to_stdout, nullptr, img.width, img.height, img.bytes_per_pixel,
                                               img.data);
        } else {
            return 0 != stbi_write_tga(filename.data(), img.width, img.height, img.bytes_per_pixel, img.data);
        }
    } break;
    }
    return true;
}

// define arguments for argparse
static void define_args(argparse::ArgumentParser& argparse) {
    argparse.add_argument("-f", "--filetype")
        .help("Types are PNG, JPEG, TGA, BMP. Derived by filename if no override given, falls back to PNG if"
              "derivation fails.")
        .nargs(1);

    argparse.add_argument("-q", "--quality")
        .help("Quality of output file in a range from 0 to 100. Only used for JPEG output.")
        .nargs(1)
        .default_value<int>(100)
        .scan<'i', int>();

    argparse.add_argument("-s", "--spread")
        .help("Spread radius in pixels for when mapping distance values to image brightness.")
        .default_value<unsigned long>(64ul)
        .scan<'i', unsigned long>();

    argparse.add_argument("-a", "--asymmetric")
        .help("SDF will be asymmetrically mapped to output. N: [-S,+S]-->[0,255]; Y: [0,S]-->[0,255]")
        .nargs(0)
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("-l", "--luminence")
        .help("SDF will be calculated from luminence chanel, as opposed to the alpha channel.")
        .nargs(0)
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("-n", "--invert")
        .help("Invert pixel value test. If set, values BELOW middle grey will be counted as \"inside\".")
        .nargs(0)
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("--list-devices")
        .help("Lists all available (and suitable) devices (if none specified, uses the default).")
        .nargs(0)
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("--device")
        .nargs(1)
        .help("Choose device by name. Use --list-devices to list all present devices. Chooses first device otherwise.");

#ifdef NDEBUG
    std::string def_log_level = "error";
#else
    std::string def_log_level = "debug";
#endif

    argparse.add_argument("--log-level")
        .help("Log level. Possible values: trace, debug, info, warning, error, critical, off.")
        .nargs(1)
        .default_value(std::move(def_log_level));

    argparse.add_argument("-i", "--input")
        .nargs(1)
        .help("Input filename. Specify \"-\" (without the quotation marks) to read from stdin.");
    argparse.add_argument("-o", "--output")
        .nargs(1)
        .help("Output filename. Specify \"-\" (without the quotation marks) to output to stdout.");
}

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::critical);

    // argument processing
    argparse::ArgumentParser argparse(argv[0]);

    define_args(argparse);

    // parse args
    try {
        argparse.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        spdlog::critical("Failed to parse arguments: {}", err.what());
        std::cout << argparse;
        return EXIT_FAILURE;
    }

    // set log level
    std::string log_level = argparse.get<std::string>("--log-level");
    std::transform(log_level.cbegin(), log_level.cend(), log_level.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    spdlog::set_level(spdlog::level::from_str(log_level));

    // sdf params
    const auto spread = argparse.get<unsigned long>("--spread");
    const auto asymmetric = argparse["--asymmetric"] == true;
    const auto use_luminence = argparse["--luminence"] == true;
    const auto invert = argparse["--invert"] == true;

    // input
    const auto opt_input = argparse.present<std::string>("--input");

    // list devices
    const auto list_devices = argparse["--list-devices"] == true;

    // input
    if (!opt_input && !list_devices) {
        spdlog::critical("Input file is required.");
        std::cout << argparse;
        return EXIT_FAILURE;
    }

    // start with vk business
    if (!ctx.init_instance()) {
        spdlog::critical("Failed to init Vulkan");
        return EXIT_FAILURE;
    }

#ifndef NDEBUG
    if (!ctx.init_debug_messenger()) {
        spdlog::critical("Failed to init debug messenger");
        return EXIT_FAILURE;
    }
#endif

    // if list-devices is specified, list physical devices and exit
    if (list_devices) {
        if (!ctx.list_vk_devices()) {
            spdlog::critical("Failed to list devices!");
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    // can begin to load image now that no other non-fatal early-exits will appear
    const auto infile = *opt_input;
    const auto fut_opt_image = std::async(open_image, infile);

    // device name
    const auto opt_device_name = argparse.present<std::string>("--device");
    if (!ctx.init_logical_device(opt_device_name)) {
        spdlog::critical("Failed to init VkDevice");
        return EXIT_FAILURE;
    }

    if (!ctx.init_command_pool()) {
        spdlog::critical("Failed to init VkCommandPool");
        return EXIT_FAILURE;
    }

    if (!ctx.init_command_buffer()) {
        spdlog::critical("Failed to init VkCommandBuffer");
        return EXIT_FAILURE;
    }

    // get fut_opt_image whenever necessary

    // more vk business goes here

    // output (filename, filetype, and quality)

    return EXIT_SUCCESS;
}

// TODO:
// - setup vk memory (selecting right type, etc, creating views or whatever)
// - compute pipeline
// - descriptor sets
// - sdf code
// - make the lower envelope parallel
// - make the envelope fill parallel
