#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb/stb_image.h>

#include <iostream>

static void open_image(std::string_view filename, std::uint8_t** data, std::size_t* width, std::size_t* height,
                       std::size_t* num_channels) {
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
    *data = static_cast<std::uint8_t*>(stbi_data);

    if (*data == nullptr) {
        spdlog::error("Loading image failed (stbi error: {})", stbi_failure_reason());
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

    auto infile = program.get<std::string>("input_file");
    auto outfile = program.get<std::string>("output_file");

    // set log level
    auto log_level = program.get<spdlog::level::level_enum>("--log-level");
    spdlog::set_level(log_level);

    // load image (from seperate thread)
    std::size_t img_w, img_h, img_channels;
    std::uint8_t* img_data;
    std::thread load_image_thread{open_image, infile, &img_data, &img_w, &img_h, &img_channels};

    // opencl setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;

    std::size_t aux_size;
    std::string aux_str;

    // opencl platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error getting platform ID (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }

    // opencl platform name
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &aux_size);
    aux_str.resize(aux_size / sizeof(char));
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, aux_str.capacity() * sizeof(char), aux_str.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error getting OpenCL platform name (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    spdlog::info("OpenCL platform name: {}", aux_str);

    // opencl platform version
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &aux_size);
    aux_str.resize(aux_size / sizeof(char));
    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, aux_str.capacity() * sizeof(char), aux_str.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error getting OpenCL platform version (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    spdlog::info("OpenCL platform version: {}", aux_str);

    // opencl device
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error getting OpenCL device ID (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }

    // opencl device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &aux_size);
    aux_str.resize(aux_size / sizeof(char));
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, aux_str.capacity() * sizeof(char), aux_str.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error getting OpenCL device name (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    spdlog::info("OpenCL device name: {}", aux_str);

    // opencl context
    ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL context (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }

    // opencl command queue
    cl_command_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        0,
    };
    queue = clCreateCommandQueueWithProperties(ctx, device, properties, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL queue (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }

    // wait on image
    load_image_thread.join();
    if (img_data == nullptr) {
        spdlog::critical("Image open failed.");
        return EXIT_FAILURE;
    }

    // free image load
    stbi_image_free(img_data);

    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
