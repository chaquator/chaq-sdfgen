#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#define CL_HPP_TARGET_OPENCL_VERSION 220
#include <CL/cl2.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

static void open_image(std::string_view filename, std::uint8_t** data, std::size_t* width, std::size_t* height,
                       std::size_t* num_channels) {
    bool use_stdin = filename == "-";

    unsigned char* stbi_data;
    int w, h, n;

    constexpr int c = 2;
    if (use_stdin) {
        spdlog::trace("Loading image from stdin");
        stbi_data = stbi_load_from_file(stdin, &w, &h, &n, c);
    } else {
        spdlog::trace("Loading image from filename \"{}\"", filename);
        stbi_data = stbi_load(filename.data(), &w, &h, &n, c);
    }
    *data = static_cast<std::uint8_t*>(stbi_data);

    if (*data == nullptr) {
        spdlog::error("Image load failed (stbi error: {})", stbi_failure_reason());
        return;
    }

    spdlog::trace("Image stats:");
    spdlog::trace("W: {}, H: {}, Channels: {}", w, h, n);

    *width = static_cast<std::size_t>(w);
    *height = static_cast<std::size_t>(h);
    *num_channels = static_cast<std::size_t>(n);
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::critical);

    // argument processing
    argparse::ArgumentParser program(argv[0], "2.0");

    program.add_argument("-f", "--filetype")
        .help("Filetype of output. Valid choices are those as supported by DevIL 1.7.8.")
        .default_value("png");

    program.add_argument("-q", "--quality")
        .help("Quality of output file in a range from 0 to 100. Only used for JPEG output.")
        .default_value(100);

    program.add_argument("-s", "--spread")
        .help("Spread radius in pixels for when mapping distance values to image brightness.")
        .default_value(64);

    program.add_argument("-a", "--asymmetric")
        .help("SDF will be asymmetrically mapped to output. N: [-S,+S]-->[0,255]; Y: [0,S]-->[0,255]")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-l", "--luminence")
        .help("SDF will be calculated from luminence chanel, as opposed to the alpha channel.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-i", "--invert")
        .help("Invert pixel value test. If set, values BELOW middle grey will be counted as \"inside\".")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--log-level")
        .help("Log level. Possible values: trace, debug, info, warn, err, critical, off.")
        .default_value("error")
        .action([](std::string value) {
            // transform to lower-case
            std::transform(value.cbegin(), value.cend(), value.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            return spdlog::level::from_str(value);
        });

    program.add_argument("input_file")
        .help("Input filename. Specify \"-\" (without the quotation marks) to read from stdin.");
    program.add_argument("output_file")
        .help("Output filename. Specify \"-\" (without the quotation marks) to output to stdout.");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        spdlog::critical("Failed to parse arguments: {}", err.what());
        std::cout << program;
        return EXIT_FAILURE;
    }

    spdlog::set_level(program.get<spdlog::level::level_enum>("--log-level"));

    auto infile = program.get<std::string>("input_file");
    auto outfile = program.get<std::string>("output_file");

    // load image (from seperate thread)
    std::size_t img_w, img_h, img_channels;
    std::uint8_t* img_data;

    std::thread load_image_thread{open_image, infile, &img_data, &img_w, &img_h, &img_channels};

    // opencl setup
    cl::Platform platform;
    cl::Device device;
    cl::Context ctx;
    cl::CommandQueue queue;

    cl_int res;
    {
        std::vector<cl::Device> devices;
        std::vector<cl::Platform> platforms;

        spdlog::trace("Getting OpenCL platforms");
        res = cl::Platform::get(&platforms);
        if (res != CL_SUCCESS) {
            spdlog::critical("Error getting platforms (OpenCL Error: {})", res);
            return EXIT_FAILURE;
        }
        platform = platforms.front();
        spdlog::info("OpenCL platform name: {}", platform.getInfo<CL_PLATFORM_NAME>());
        spdlog::info("OpenCL platform version: {}", platform.getInfo<CL_PLATFORM_VERSION>());

        spdlog::trace("Grabbing device from platform");
        res = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (res != CL_SUCCESS) {
            spdlog::critical("Error getting device from platform (OpenCL Error: {})", res);
            return EXIT_FAILURE;
        }
        device = devices.front();
        spdlog::info("OpenCL device name: {}", device.getInfo<CL_DEVICE_NAME>());

        spdlog::trace("Creating context with device");
        ctx = cl::Context(device, NULL, NULL, NULL, &res);
        if (res != CL_SUCCESS) {
            spdlog::critical("Error creating context (OpenCL Error: {})", res);
            return EXIT_FAILURE;
        }
        spdlog::trace("Context created");
    }

    spdlog::trace("Creating command queue");
    queue = cl::CommandQueue(ctx, device, 0, &res);
    if (res != CL_SUCCESS) {
        spdlog::critical("Error creating command queue (OpenCL Error: {})", res);
        return EXIT_FAILURE;
    }

    // wait on image
    load_image_thread.join();
    if (img_data == nullptr) {
        spdlog::error("Image open failed.");
        return EXIT_FAILURE;
    }

    // free image load
    stbi_image_free(img_data);

    return 0;
}
