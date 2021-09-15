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
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// small helper class that gives raii semantics for trivial handles that are already acquired
template <typename T, typename F>
class auto_release {
  private:
    T m_handle{};
    F m_release_func{};
    bool m_valid{false};

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
    cl_ulong bytes_per_pixel = 2;
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
        return {};
    }

    spdlog::trace("Image stats:");
    spdlog::trace("W: {}, H: {}, Channels: {}", w, h, n);

    return stbi_img{
        static_cast<cl_uchar*>(stbi_data),
        static_cast<cl_ulong>(w),
        static_cast<cl_ulong>(h),
    };
}

// based on code from https://insanecoding.blogspot.com/2011/11/how-to-read-in-file-in-c.html
static std::optional<std::string> get_file_contents(const char* filename) {
    spdlog::trace("Opening file {}", filename);
    std::ifstream in_file{filename, std::ios_base::in | std::ios_base::binary};

    if (!in_file) {
        spdlog::warn("Failed to open file \"{}\"", filename);
        return {};
    }

    std::string contents;

    // fit string to size of file contents
    in_file.seekg(0, std::ios::end);
    contents.resize(in_file.tellg());
    in_file.seekg(0, std::ios::beg);

    // read in data
    in_file.read(contents.data(), contents.size());

    return contents;
}

static std::optional<std::vector<cl_platform_id>> get_platforms() {
    cl_int err;
    spdlog::trace("Listing platforms");

    // get number of platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS) {
        spdlog::warn("Error listing OpenCL platforms (OpenCL error: {})", err);
        return {};
    }

    // fill out platforms
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::warn("Error getting OpenCL platforms (OpenCL error: {})", err);
        return {};
    }

    return platforms;
}

static std::optional<std::string> get_platform_name(cl_platform_id platform) {
    cl_int err;

    // platform name size
    std::size_t plat_name_size;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &plat_name_size);
    if (err != CL_SUCCESS) {
        spdlog::warn("Error getting OpenCL platform name for {} (OpenCL error: {})", static_cast<void*>(platform), err);
        return {};
    }

    // platform name data
    std::string plat_name;
    plat_name.resize(plat_name_size - 1);
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, plat_name.capacity(), plat_name.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::warn("Error getting OpenCL platform name for {} (OpenCL error: {})", static_cast<void*>(platform), err);
        return {};
    }

    return plat_name;
}

static std::optional<std::vector<cl_device_id>> get_devices(cl_platform_id platform) {
    cl_int err;
    spdlog::trace("Listing devices for platform {}", static_cast<void*>(platform));

    // get number of devices
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (err != CL_SUCCESS) {
        spdlog::warn("Error listing OpenCL devices (OpenCL error: {})", err);
        return {};
    }

    // fill out devices
    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::warn("Error getting OpenCL devices (OpenCL error: {})", err);
        return {};
    }

    return devices;
}

static std::optional<std::string> get_device_name(cl_device_id device) {
    cl_int err;

    // device name size
    std::size_t device_name_size;
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_size);
    if (err != CL_SUCCESS) {
        spdlog::warn("Error getting OpenCL device name for {} (OpenCL error: {})", static_cast<void*>(device), err);
        return {};
    }

    // device name data
    std::string device_name;
    device_name.resize(device_name_size - 1);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, device_name.capacity(), device_name.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::warn("Error getting OpenCL device name for {} (OpenCL error: {})", static_cast<void*>(device), err);
        return {};
    }

    return device_name;
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
        .default_value<cl_ulong>(64ul)
        .scan<'i', cl_ulong>();

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

    argparse.add_argument("--list-devices")
        .help("Lists all devices on a platform (if none specified, uses the default).")
        .nargs(0)
        .default_value(false)
        .implicit_value(true);

    argparse.add_argument("--device")
        .nargs(1)
        .help("Choose device by name. Use --list-devices to list the device for a platform. Chooses first device "
              "otherwise.");

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

    // if list-platforms or list-devices is specified, process when appropriate and then exit
    bool list_platforms = argparse["--list-platforms"] == true;
    bool list_devices = argparse["--list-devices"] == true;

    if (list_platforms) {
        const auto platforms_opt = get_platforms();
        if (!platforms_opt) {
            spdlog::critical("Could not get OpenCL platforms");
            return EXIT_FAILURE;
        }

        const auto& platforms = *platforms_opt;
        for (const auto& p : platforms) {
            const auto name_opt = get_platform_name(p);
            if (!name_opt) {
                spdlog::warn("Failed to get OpenCL platform name of {}, skipping.", static_cast<void*>(p));
                continue;
            }

            const auto& name = *name_opt;
            std::cout << name << '\n';
        }

        return EXIT_SUCCESS;
    }

    // opencl setup
    cl_int err;

    std::size_t aux_size{};
    std::string aux_str{};

    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // opencl platform
    spdlog::trace("Getting platforms");
    const auto platform_arg_opt = argparse.present<std::string>("--platform");
    if (platform_arg_opt) {
        // find platform by name
        spdlog::trace("Searching for platform by name.");
        const auto platforms_opt = get_platforms();
        if (!platforms_opt) {
            spdlog::critical("Could not get OpenCL platforms");
            return EXIT_FAILURE;
        }
        const auto& platforms = *platforms_opt;

        const auto& desired_platform_name = *platform_arg_opt;
        spdlog::trace("Looking for OpenCL platform with name \"{}\"", desired_platform_name);

        // find platform with desired name
        const auto name_find =
            std::find_if(platforms.cbegin(), platforms.cend(), [&desired_platform_name](const auto& p) {
                auto p_name_opt = get_platform_name(p);
                if (!p_name_opt) {
                    spdlog::warn("Failed to get OpenCL platform name for {}, skipping.", static_cast<void*>(p));
                    return false;
                }
                const auto& p_name = *p_name_opt;

                spdlog::trace("Looking at OpenCL platform \"{}\"", p_name);

                const auto find_pos = p_name.find(desired_platform_name);
                spdlog::trace("\"{}\".find(\"{}\"): {}", p_name, desired_platform_name, find_pos);
                return find_pos != std::string::npos;
            });

        if (name_find == platforms.cend()) {
            spdlog::critical("Could not find OpenCL platform with name \"{}\"", desired_platform_name);
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
        spdlog::error("Could not get plaform name");
    } else {
        spdlog::info("OpenCL platform name: {}", *plat_name_opt);
    }

    // opencl platform version
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &aux_size);
    aux_str.resize(aux_size - 1);
    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, aux_str.capacity() * sizeof(char), aux_str.data(), nullptr);
    if (err != CL_SUCCESS) {
        spdlog::error("Error getting OpenCL platform version (OpenCL error: {})", err);
        return EXIT_FAILURE;
    } else {
        spdlog::info("OpenCL platform version: {}", aux_str);
    }

    // opencl device
    if (list_devices) {
        const auto devices_opt = get_devices(platform);
        if (!devices_opt) {
            spdlog::critical("Could not get OpenCL devices");
            return EXIT_FAILURE;
        }

        const auto& devices = *devices_opt;
        for (const auto& d : devices) {
            const auto name_opt = get_device_name(d);
            if (!name_opt) {
                spdlog::warn("Failed to get OpenCL device name of {}, skipping.", static_cast<void*>(d));
                continue;
            }

            const auto& name = *name_opt;
            std::cout << name << '\n';
        }

        return EXIT_SUCCESS;
    }

    // knowing that neither --list-devices or --list-platforms is called, now is a good time
    // to begin loading the resources asynchronously

    // first check for both input and output
    auto input_opt = argparse.present<std::string>("--input");
    if (!input_opt) {
        spdlog::critical("Input file is required");
        std::cout << argparse;
        return EXIT_FAILURE;
    }
    const auto& infile = *input_opt;

    auto output_opt = argparse.present<std::string>("--output");
    if (!output_opt) {
        spdlog::critical("Output file is required");
        std::cout << argparse;
        return EXIT_FAILURE;
    }
    const auto& outfile = *output_opt;

    // load image
    auto image_fut = std::async(open_image, infile);

    // load source content
    auto source_fut = std::async(get_file_contents, "sdf.cl");

    // opencl device
    auto device_arg_opt = argparse.present<std::string>("--device");
    if (device_arg_opt) {
        // find device by name
        spdlog::trace("Searching for device by name.");
        const auto devices_opt = get_devices(platform);
        if (!devices_opt) {
            spdlog::critical("Could not get OpenCL devices.");
            return EXIT_FAILURE;
        }
        const auto& devices = *devices_opt;

        const auto& desired_device_name = *device_arg_opt;
        spdlog::trace("Looking for OpenCL device with name \"{}\"", desired_device_name);

        // find device with desired name
        const auto name_find = std::find_if(devices.cbegin(), devices.cend(), [&desired_device_name](const auto& d) {
            const auto d_name_opt = get_device_name(d);
            if (!d_name_opt) {
                spdlog::warn("Failed to get OpenCL deivce name for {}, skipping.", static_cast<void*>(d));
                return false;
            }
            const auto& d_name = *d_name_opt;
            spdlog::trace("Looking at OpenCL device \"{}\"", d_name);

            const auto find_pos = d_name.find(desired_device_name);
            spdlog::trace("\"{}\".find(\"{}\"): {}", d_name, desired_device_name, find_pos);
            return find_pos != std::string::npos;
        });

        if (name_find == devices.cend()) {
            spdlog::critical("Could not find OpenCL device with name \"{}\"", desired_device_name);
            return EXIT_FAILURE;
        }

        device = *name_find;
    } else {
        // get first device
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            spdlog::critical("Error getting OpenCL device ID (OpenCL error: {})", err);
            return EXIT_FAILURE;
        }
    }
    spdlog::trace("Got OpenCL device");

    // opencl device name
    const auto name_opt = get_device_name(device);
    if (!name_opt) {
        spdlog::error("Failed to get OpenCL device name");
    } else {
        spdlog::info("OpenCL device name: {}", *name_opt);
    }

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

    // get source string
    spdlog::trace("Waiting on source string");
    auto source_opt = source_fut.get();
    if (!source_opt) {
        spdlog::critical("Could not find source file.");
        return EXIT_FAILURE;
    }
    const char* src = source_opt->data();
    const std::size_t len = source_opt->length();
    spdlog::trace("Got source string");

    // opencl source program
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

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &aux_size);
    if (aux_size > 2) {
        aux_str.resize(aux_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, aux_str.capacity() * sizeof(char), aux_str.data(),
                              nullptr);
        spdlog::info("Build log: {}", aux_str);
    }

    // opencl kernel
    auto kernel_name = "sdf";
    kernel = clCreateKernel(program, kernel_name, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Error creating OpenCL kernel (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release kernel_release{kernel, clReleaseKernel};
    spdlog::trace("Created OpenCL kernel \"{}\"", kernel_name);

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

    // opencl image
    cl_image_format img_fmt;
    img_fmt.image_channel_order = CL_RA;
    img_fmt.image_channel_data_type = CL_UNSIGNED_INT8;
    cl_image_desc img_dsc;
    img_dsc.image_type = CL_MEM_OBJECT_IMAGE2D;
    img_dsc.image_width = image.width;
    img_dsc.image_height = image.height;
    img_dsc.image_row_pitch = image.width * image.bytes_per_pixel;
    img_dsc.num_mip_levels = 0;
    img_dsc.num_samples = 0;
    img_dsc.buffer = nullptr;
    cl_mem image_mem = clCreateImage(ctx, CL_MEM_READ_WRITE, &img_fmt, &img_dsc, nullptr, &err);
    if (err != CL_SUCCESS) {
        spdlog::critical("Failed to create OpenCL image (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    auto_release image_mem_release{image_mem, clReleaseMemObject};

    // opencl kernel arguments
    cl_ulong spread = argparse.get<cl_ulong>("--spread");
    cl_char use_luminence = (cl_uchar)(argparse["--luminence"] == true);
    cl_uchar invert = (cl_uchar)(argparse["--invert"] == true);
    cl_uchar asymmetric = (cl_uchar)(argparse["--asymmetric"] == true);
    spdlog::trace("Spread: {}", spread);
    spdlog::trace("Use luminence: {}", use_luminence);
    spdlog::trace("Invert: {}", invert);
    spdlog::trace("Asymmetric: {}", asymmetric);

    // 0 - img
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_mem);
    if (err != CL_SUCCESS) {
        spdlog::critical("Setting kernel arguments went wrong {}", __LINE__);
        return EXIT_FAILURE;
    }
    // 1 - spread
    err = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &spread);
    if (err != CL_SUCCESS) {
        spdlog::critical("Setting kernel arguments went wrong {}", __LINE__);
        return EXIT_FAILURE;
    }
    // 2 - use_luminence
    err = clSetKernelArg(kernel, 2, sizeof(cl_uchar), &use_luminence);
    if (err != CL_SUCCESS) {
        spdlog::critical("Setting kernel arguments went wrong {}", __LINE__);
        return EXIT_FAILURE;
    }
    // 3 - invert
    err = clSetKernelArg(kernel, 3, sizeof(cl_uchar), &invert);
    if (err != CL_SUCCESS) {
        spdlog::critical("Setting kernel arguments went wrong {}", __LINE__);
        return EXIT_FAILURE;
    }
    // 4 - asymmetric
    err = clSetKernelArg(kernel, 4, sizeof(cl_uchar), &asymmetric);
    if (err != CL_SUCCESS) {
        spdlog::critical("Setting kernel arguments went wrong {}", __LINE__);
        return EXIT_FAILURE;
    }

    // opencl enqueues
    size_t img_origin[3] = {0, 0, 0};
    size_t img_region[3] = {image.width, image.height, 1};
    size_t work_size[2] = {image.width, image.height};

    cl_event img_write_evt;
    cl_event kernel_evt;

    // image write
    err = clEnqueueWriteImage(queue, image_mem, CL_FALSE, img_origin, img_region, image.width * image.bytes_per_pixel,
                              0, image.data, 0, nullptr, &img_write_evt);
    if (err != CL_SUCCESS) {
        spdlog::critical("Failed to enqueue image write (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    // kernel execution
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, work_size, nullptr, 1, &img_write_evt, &kernel_evt);
    if (err != CL_SUCCESS) {
        spdlog::critical("Failed to enqueue kernel execution (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    // image read back
    err = clEnqueueReadImage(queue, image_mem, CL_FALSE, img_origin, img_region, image.width * image.bytes_per_pixel, 0,
                             image.data, 1, &kernel_evt, nullptr);
    if (err != CL_SUCCESS) {
        spdlog::critical("Failed to enqueue image read back (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }

    // opencl wait
    spdlog::trace("Waiting on queue");
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        spdlog::critical("Erorr finishing queue (OpenCL error: {})", err);
        return EXIT_FAILURE;
    }
    spdlog::trace("Queue finished");

    // std::cout.write((const char*)image.data, image.width * image.height * image.bytes_per_pixel);
    /*
    for (cl_ulong y = 0; y < image.height; ++y) {
        for (cl_ulong x = 0; x < image.width; ++x) {
            std::cout << *(image.data + image.bytes_per_pixel * (x + image.width * y));
        }
    }*/

    // write back file
    stbi_write_png("out.png", image.width, image.height, image.bytes_per_pixel, image.data,
                   (image.width * image.bytes_per_pixel));

    return EXIT_SUCCESS;
}

// TODO: logging pass, organize pass
