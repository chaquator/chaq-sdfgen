#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb/stb_image.h>

extern const char _binary_opencl_sdf_cl_start, _binary_opencl_sdf_cl_end;

// small helper class that gives raii semantics for trivial handles that are already acquired
template <typename T, typename F>
class auto_release {
  private:
    T handle = {};
    F release_func = {};
    bool valid = false;

  public:
    auto_release() = default;
    auto_release(T h, F f) : handle(h), release_func(f), valid(true) {}
    auto_release(auto_release<T, F>&) = delete;
    auto_release(auto_release<T, F>&&) = default;
    ~auto_release() {
        if (valid) release_func(handle);
    }
};

struct stbi_img {
    std::uint8_t* data = nullptr;
    std::size_t width = 0;
    std::size_t height = 0;
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
        spdlog::error("Loading image failed (stbi error: {})", stbi_failure_reason());
        return {};
    }

    spdlog::trace("Image stats:");
    spdlog::trace("W: {}, H: {}, Channels: {}", w, h, n);

    return {{
        static_cast<std::uint8_t*>(stbi_data),
        static_cast<std::size_t>(w),
        static_cast<std::size_t>(h),
    }};
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::critical);

    // argument processing
    argparse::ArgumentParser argparse(argv[0], "2.0");

    argparse.add_argument("-f", "--filetype")
        .help("Filetype of output. Supoprted types are PNG, JPEG, TGA, BMP")
        .default_value("png");

    argparse.add_argument("-q", "--quality")
        .help("Quality of output file in a range from 0 to 100. Only used for JPEG output.")
        .default_value(100);

    argparse.add_argument("-s", "--spread")
        .help("Spread radius in pixels for when mapping distance values to image brightness.")
        .default_value(64);

    argparse.add_argument("-a", "--asymmetric")
        .help("SDF will be asymmetrically mapped to output. N: [-S,+S]-->[0,255]; Y: [0,S]-->[0,255]")
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("-l", "--luminence")
        .help("SDF will be calculated from luminence chanel, as opposed to the alpha channel.")
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("-i", "--invert")
        .help("Invert pixel value test. If set, values BELOW middle grey will be counted as \"inside\".")
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("--log-level")
        .help("Log level. Possible values: trace, debug, info, warn, err, critical, off.")
        .default_value("error")
        .action([](std::string value) {
            // transform to lower-case
            std::transform(value.cbegin(), value.cend(), value.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            return spdlog::level::from_str(value);
        });

    argparse.add_argument("input_file")
        .help("Input filename. Specify \"-\" (without the quotation marks) to read from stdin.");
    argparse.add_argument("output_file")
        .help("Output filename. Specify \"-\" (without the quotation marks) to output to stdout.");

    try {
        argparse.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        spdlog::critical("Failed to parse arguments: {}", err.what());
        std::cout << argparse;
        return EXIT_FAILURE;
    }

    auto infile = argparse.get<std::string>("input_file");
    auto outfile = argparse.get<std::string>("output_file");

    // set log level
    auto log_level = argparse.get<spdlog::level::level_enum>("--log-level");
    spdlog::set_level(log_level);

    // load image asynchronously
    auto image_fut = std::async(open_image, infile);

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
    auto_release ctx_release{ctx, clReleaseContext};

    // opencl command queue
    cl_command_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
        //
        0,
    };
    queue = clCreateCommandQueueWithProperties(ctx, device, properties, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL queue (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release queue_release{queue, clReleaseCommandQueue};

    // wait on image
    auto image_opt = image_fut.get();
    if (!image_opt) {
        spdlog::critical("Image open failed.");
        return EXIT_FAILURE;
    }
    auto image_s = image_opt.value();
    auto_release image_release{image_s.data, stbi_image_free};

    // opencl program
    const char* sources[] = {
        &_binary_opencl_sdf_cl_start,
        nullptr,
    };
    const std::size_t lengths[] = {
        static_cast<std::size_t>(&_binary_opencl_sdf_cl_end - &_binary_opencl_sdf_cl_start),
        0,
    };
    cl_program program = clCreateProgramWithSource(ctx, 1, sources, lengths, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL program (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release program_release{program, clReleaseProgram};
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error building OpenCL program (OpenCL error: {})", err);
        // if (err == CL_BUILD_PROGRAM_FAILURE) {
        if (true) {
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &aux_size);
            aux_str.resize(aux_size / sizeof(char));
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, aux_str.capacity() * sizeof(char),
                                  aux_str.data(), nullptr);
            spdlog::info("Build log: {}", aux_str);
        }
        return EXIT_FAILURE;
    }

    // opencl kernel
    cl_kernel kernel = clCreateKernel(program, "testy", &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL kernel (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release kernel_release{kernel, clReleaseKernel};

    std::size_t work_size = 1024;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error enqueueing NDRange for OpenCL kernel (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error finishing OpenCL queue (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }

    // opencl buffers
    // one for the original image
    // one for inside, one for outside
    // auxiliary buffers for both
    // help me

    // opencl args for inside

    // opencl args for outside

    // enqueues

    // finish

    return 0;
}
