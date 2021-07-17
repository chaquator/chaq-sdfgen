#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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

// small helper class that gives raii semantics for trivial handles that are already acquired
template <typename T, typename F>
class auto_release {
  private:
    T hndl = {};
    F release_func = {};
    bool valid = false;

  public:
    const T handle() const { return hndl; }
    auto_release() = default;
    auto_release(T h, F f) : hndl(h), release_func(f), valid(true) {}
    auto_release(auto_release<T, F>&) = delete;
    auto_release(auto_release<T, F>&&) = default;
    ~auto_release() {
        if (valid) release_func(hndl);
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

// based on code from https://insanecoding.blogspot.com/2011/11/how-to-read-in-file-in-c.html
static std::optional<std::string> get_file_contents(const char* filename) {
    std::ifstream in{filename, std::ios_base::in | std::ios_base::binary};

    if (!in) {
        spdlog::error("Failed to open file \"{}\"", filename);
        return {};
    }

    std::string contents;

    in.seekg(0, std::ios::end);
    contents.resize(in.tellg());
    in.seekg(0, std::ios::beg);

    in.read(contents.data(), contents.size());

    return contents;
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
        .help("Log level. Possible values: trace, debug, info, warning, error, critical, off.")
        .default_value(std::string("error"));

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

    auto outfile = argparse.get<std::string>("output_file");

    // set log level
    std::string log_level = argparse.get<std::string>("--log-level");
    std::transform(log_level.cbegin(), log_level.cend(), log_level.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    spdlog::set_level(spdlog::level::from_str(log_level));

    // load image asynchronously
    auto infile = argparse.get<std::string>("input_file");
    auto image_fut = std::async(open_image, infile);

    // load shader content asynchronously
    auto shader_fut = std::async(get_file_contents, "sdf.cl");

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
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
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

    // get shader string
    auto shader_opt = shader_fut.get();
    if (!shader_opt) {
        spdlog::critical("Could not find shader file.");
        return EXIT_FAILURE;
    }
    auto shader_str = shader_opt.value();

    // opencl shader program
    const char* src = shader_str.data();
    const std::size_t len = shader_str.length();
    cl_program program = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL program (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release program_release{program, clReleaseProgram};
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error building OpenCL program (OpenCL error: {})", err);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &aux_size);
        aux_str.resize(aux_size / sizeof(char));
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, aux_str.capacity() * sizeof(char), aux_str.data(),
                              nullptr);
        spdlog::info("Build log: {}", aux_str);
        return EXIT_FAILURE;
    }

    // opencl kernel
    cl_kernel kernel = clCreateKernel(program, "testy", &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL kernel (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release kernel_release{kernel, clReleaseKernel};

    // wait on image
    auto image_opt = image_fut.get();
    if (!image_opt) {
        spdlog::critical("Image open failed.");
        return EXIT_FAILURE;
    }
    auto image_s = image_opt.value();
    auto_release image_release{image_s.data, stbi_image_free};

    /*
    std::size_t work_size = 2048;
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
    */

    // TODO: skip task flow for inside image if asymmetric flag is set

    // opencl buffers
    std::size_t img_size_base = image_s.height * image_s.width;
    std::size_t img_size_bytes = img_size_base * sizeof(std::uint8_t);
    std::size_t max_dim = image_s.height > image_s.width ? image_s.height : image_s.width;
    std::size_t work_group_size;

    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(std::size_t), &work_group_size,
                                   nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Failed to query kernel work group size (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }

    // buffers declared for each side
    // image buffer -- image_size_bytes
    // intermediate transpose float image -- image_size_base * sizeof(float)
    // output distance function float image -- image_size_base * sizeof(float)
    // aux vertices buffer -- max_dim * work group size * sizeof(std::size_t)
    // aux vertex height buffer -- max_dim * work group size * sizeof(float)
    // aux break point buffer -- (max_dim-1) * work group size * sizeof(std::size_t)

    // after computing both distance functions, will consolidate the results and
    // write back to image buffer for outside

    // outside
    // image buffer
    cl_mem img_buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, img_size_bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside image buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release img_buffer_release{img_buffer, clReleaseMemObject};
    // intermediate transpose flaot image
    cl_mem intermediate_transpose =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, img_size_base * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside image intermediate transpose buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release intermediate_transpose_release{intermediate_transpose, clReleaseMemObject};
    // output distance function float image
    cl_mem dist_func = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, img_size_base * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside distance function output buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release dist_func_release{dist_func, clReleaseMemObject};
    // aux vertices buffer
    cl_mem vertices =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, max_dim * work_group_size * sizeof(std::size_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside vertices buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release vertices_release{vertices, clRetainMemObject};
    // aux vertex height buffer
    cl_mem vert_height =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, max_dim * work_group_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside vertex height buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release vert_height_release{vert_height, clReleaseMemObject};
    // aux break point buffer
    cl_mem breakpoint =
        clCreateBuffer(ctx, CL_MEM_READ_WRITE, (max_dim - 1) * work_group_size * sizeof(std::size_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside breakpoint buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release breakpoint_release{breakpoint, clReleaseMemObject};

    // opencl arguments

    // enqueues
    // buffer write
    // taskflow
    // buffer read back

    // finish

    // TODO: once taskflow for distance transform works for at least one side, wrap up into struct and function
    // to make repeating code for other side easier

    return EXIT_SUCCESS;
}
