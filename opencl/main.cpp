#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

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
    T m_handle = {};
    F m_release_func = {};
    bool m_valid = false;

  public:
    const T handle() const { return m_handle; }
    const F release_function() const { return m_release_func; }
    bool valid() const { return m_valid; }

    void invalidate() { m_valid = false; }

    auto_release() = default;
    auto_release(T h, F f) : m_handle(h), m_release_func(f), m_valid(true) {}
    auto_release(auto_release<T, F>&& other) {
        std::swap(m_handle, other.m_handle);
        std::swap(m_release_func, other.m_release_func);
        std::swap(m_valid, other.m_valid);
    }
    auto_release<T, F>& operator=(auto_release<T, F>&& other) {
        std::swap(m_handle, other.m_handle);
        std::swap(m_release_func, other.m_release_func);
        std::swap(m_valid, other.m_valid);
        return *this;
    };
    auto_release(auto_release<T, F>&) = delete;
    auto_release<T, F>& operator=(const auto_release<T, F>&) = delete;
    ~auto_release() {
        if (m_valid) m_release_func(m_handle);
    }
};

struct stbi_img {
    cl_uchar* data = nullptr;
    cl_ulong width = 0;
    cl_ulong height = 0;
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
        static_cast<cl_uchar*>(stbi_data),
        static_cast<cl_ulong>(w),
        static_cast<cl_ulong>(h),
    }};
}

// based on code from https://insanecoding.blogspot.com/2011/11/how-to-read-in-file-in-c.html
static std::optional<std::string> get_file_contents(const char* filename) {
    spdlog::trace("Opening file {}", filename);
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

    return std::make_optional(std::move(contents));
}

static std::optional<std::vector<cl_platform_id>> get_platforms() {
    cl_int err;
    spdlog::trace("Listing platforms");
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS) {
        spdlog::error("Error listing OpenCL platforms (OpenCL error: {})", err);
        return {};
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::error("Error listing OpenCL platforms (OpenCL error: {})", err);
        return {};
    }
    return std::make_optional(std::move(platforms));
}

static std::optional<std::string> get_platform_name(cl_platform_id platform) {
    cl_int err;
    std::size_t plat_name_size;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &plat_name_size);
    if (err != CL_SUCCESS) {
        spdlog::error("Error getting OpenCL platform name for {} (OpenCL error: {})", static_cast<void*>(platform),
                      err);
        return {};
    }
    std::string plat_name;
    plat_name.resize(plat_name_size);
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, plat_name.capacity(), plat_name.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::error("Error getting OpenCL platform name for {} (OpenCL error: {})", static_cast<void*>(platform),
                      err);
        return {};
    }
    return std::make_optional(std::move(plat_name));
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::critical);

    // argument processing
    argparse::ArgumentParser argparse(argv[0], "2.0");

    argparse.add_argument("-f", "--filetype")
        .help("Filetype of output. Supoprted types are PNG, JPEG, TGA, BMP")
        .nargs(1)
        .default_value("png");

    argparse.add_argument("-q", "--quality")
        .help("Quality of output file in a range from 0 to 100. Only used for JPEG output.")
        .nargs(1)
        .default_value(100);

    argparse.add_argument("-s", "--spread")
        .help("Spread radius in pixels for when mapping distance values to image brightness.")
        .nargs(1)
        .default_value(64);

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

    argparse.add_argument("--list-platforms")
        .help("List all platforms on machine by name then exits.")
        .nargs(0)
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("--platform")
        .nargs(1)
        .help(
            "Choose platform by name. Use --list-platforms to view platform names. Chooses first platform otherwise.");

    argparse.add_argument("--log-level")
        .help("Log level. Possible values: trace, debug, info, warning, error, critical, off.")
        .nargs(1)
        .default_value(std::string("error"));

    argparse.add_argument("-i", "--input")
        .nargs(1)
        .help("Input filename. Specify \"-\" (without the quotation marks) to read from stdin.");
    argparse.add_argument("-o", "--output")
        .nargs(1)
        .help("Output filename. Specify \"-\" (without the quotation marks) to output to stdout.");

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

    // if list-platforms or list-devices is specified, process accordingly and then exit
    if (argparse["--list-platforms"] == true) {
        auto platforms_opt = get_platforms();
        if (!platforms_opt) {
            spdlog::critical("Could not get OpenCL platforms");
            return EXIT_FAILURE;
        }
        auto& platforms = *platforms_opt;
        for (const auto& p : platforms) {
            auto name_opt = get_platform_name(p);
            if (!name_opt) {
                spdlog::error("Failed to get OpenCL platform name of {}, skipping.", static_cast<void*>(p));
                continue;
            }
            auto& name = *name_opt;
            std::cout << name << '\n';
        }

        return EXIT_SUCCESS;
    }

    // load image asynchronously
    auto input_opt = argparse.present<std::string>("--input");
    if (!input_opt) {
        spdlog::critical("Input file is required");
        return EXIT_FAILURE;
    }
    const auto& infile = *input_opt;
    auto image_fut = std::async(open_image, infile);

    // load shader content asynchronously
    auto shader_fut = std::async(get_file_contents, "sdf.cl");

    // opencl setup
    cl_int err;
    cl_platform_id platform;

    std::size_t aux_size;
    std::string aux_str;

    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // opencl platform
    spdlog::trace("Getting platforms");
    auto platform_opt = argparse.present<std::string>("--platform");
    if (platform_opt) {
        // find platform by name
        spdlog::trace("Searching for platform by name.");
        auto plats_opt = get_platforms();
        if (!plats_opt) {
            spdlog::critical("Could not get OpenCL platforms");
            return EXIT_FAILURE;
        }
        auto& platforms = *plats_opt;

        const auto& plat_name = *platform_opt;
        spdlog::trace("Looking for OpenCL platform with name \"{}\"", plat_name);

        auto name_find = std::find_if(platforms.cbegin(), platforms.cend(), [&plat_name](const auto& p) {
            auto p_name_opt = get_platform_name(p);
            if (!p_name_opt) {
                spdlog::error("Failed to get OpenCL platform name for {}, skipping.", static_cast<void*>(p));
                return false;
            }
            const auto& p_name = *p_name_opt;

            spdlog::trace("Looking at OpenCL platform \"{}\"", p_name);

            auto find_pos = p_name.find(plat_name);
            spdlog::trace("\"{}\".find(\"{}\"): {}", p_name, plat_name, find_pos);
            return find_pos != std::string::npos;
        });
        if (name_find == platforms.cend()) {
            spdlog::critical("Could not find OpenCL platform with name \"{}\"", plat_name);
            return EXIT_FAILURE;
        }

        platform = *name_find;
    } else {
        // get first available platform
        spdlog::trace("Getting first available platform.");
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            spdlog::critical("Error getting platform ID (OpenCL error: {})", err);
            return EXIT_FAILURE;
        }
    }
    spdlog::trace("Got OpenCL platform");

    // opencl platform name
    auto plat_name_opt = get_platform_name(platform);
    if (!plat_name_opt) {
        spdlog::critical("Could not get plaform name");
        return EXIT_FAILURE;
    }
    spdlog::info("OpenCL platform name: {}", *plat_name_opt);

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
    spdlog::trace("Got OpenCL device");

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
    spdlog::trace("Created OpenCL context");

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
    spdlog::trace("Created OpenCL command queue");

    // get shader string
    spdlog::trace("Waiting on shader string");
    auto shader_opt = shader_fut.get();
    if (!shader_opt) {
        spdlog::critical("Could not find shader file.");
        return EXIT_FAILURE;
    }
    const char* src = shader_opt->data();
    const std::size_t len = shader_opt->length();
    spdlog::trace("Got shader string");

    // opencl shader program
    program = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL program (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release program_release{program, clReleaseProgram};
    spdlog::trace("Created OpenCL program");
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
    spdlog::trace("Built OpenCL program");

    // opencl kernel
    kernel = clCreateKernel(program, "dist_transform_part1", &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL kernel (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release kernel_release{kernel, clReleaseKernel};
    spdlog::trace("Created OpenCL kernel");

    // wait on image
    spdlog::trace("Waiting on image data");
    auto image_opt = image_fut.get();
    if (!image_opt) {
        spdlog::critical("Image open failed.");
        return EXIT_FAILURE;
    }
    auto image = *image_opt;
    auto_release image_release{image.data, stbi_image_free};
    spdlog::trace("Got image data");

    // TODO: skip task flow for inside image if asymmetric flag is set

    // opencl buffers
    std::size_t img_size_base = image.height * image.width;
    std::size_t img_size_bytes = img_size_base * sizeof(cl_uchar);
    spdlog::trace("Image size in bytes: {}", img_size_bytes);

    // buffers declared for each side
    // image buffer -- image_size_bytes
    // intermediate transpose float image -- image_size_base * sizeof(float)
    // output distance function float image -- image_size_base * sizeof(float)
    // aux vertices buffer -- max_dim * sizeof(std::size_t)
    // aux vertex height buffer -- max_dim * sizeof(float)
    // aux break point buffer -- (max_dim-1) * sizeof(float)

    // after computing both distance functions, will consolidate the results and
    // write back to image buffer for outside

    cl_mem img_buffer;
    cl_mem intermediate_transpose;
    // cl_mem dist_func;

    // outside
    // image buffer
    spdlog::trace("Creating OpenCL buffers");
    img_buffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, img_size_bytes, nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside image buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release img_buffer_release{img_buffer, clReleaseMemObject};
    // intermediate transpose flaot image
    intermediate_transpose = clCreateBuffer(ctx, CL_MEM_READ_WRITE, img_size_base * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside image intermediate transpose buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release intermediate_transpose_release{intermediate_transpose, clReleaseMemObject};
    /*
    // output distance function float image
    dist_func = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, img_size_base * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Could not create outside distance function output buffer (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release dist_func_release{dist_func, clReleaseMemObject};
    */

    // opencl arguments
    cl_ulong bytes_per_pixel = 2;

    // TODO: decide w param
    cl_ulong channel_offset = 1;
    cl_uchar invert = 0;

    std::size_t aux_base_size = image.width;

    // width
    clSetKernelArg(kernel, 0, sizeof(cl_ulong), &image.width);
    // height
    clSetKernelArg(kernel, 1, sizeof(cl_ulong), &image.height);
    // bytes_per_pixel
    clSetKernelArg(kernel, 2, sizeof(cl_ulong), &bytes_per_pixel);
    // channel_offset
    clSetKernelArg(kernel, 3, sizeof(cl_ulong), &channel_offset);
    // invert
    clSetKernelArg(kernel, 4, sizeof(cl_uchar), &invert);
    // in_image
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &img_buffer);
    // out_image
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &intermediate_transpose);
    // aux_vertices
    clSetKernelArg(kernel, 7, aux_base_size * sizeof(cl_ulong), nullptr);
    // aux_vertex_heights
    clSetKernelArg(kernel, 8, aux_base_size * sizeof(cl_float), nullptr);
    // aux_breakpoints
    clSetKernelArg(kernel, 9, (aux_base_size - 1) * sizeof(cl_float), nullptr);

    // enqueues
    // buffer write
    cl_event img_buffer_write;
    clEnqueueWriteBuffer(queue, img_buffer, CL_FALSE, 0, img_size_bytes, image.data, 0, nullptr, &img_buffer_write);
    // taskflow
    cl_event kernel_exec;
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &image.height, nullptr, 1, &img_buffer_write, &kernel_exec);
    // buffer read back
    // read back intermediate_transpose for testing
    std::vector<cl_uchar> test_out_img(img_size_base);
    clEnqueueReadBuffer(queue, intermediate_transpose, CL_FALSE, 0, img_size_bytes, test_out_img.data(), 1,
                        &kernel_exec, nullptr);

    // finish
    clFinish(queue);

    // TODO: write out image for comparison

    // TODO: once taskflow for distance transform works for at least one side, wrap up into struct and function
    // to make repeating code for other side easier

    return EXIT_SUCCESS;
}
