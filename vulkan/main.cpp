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

#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "filetype.h"

static const char* app_name = "chaq_sdfgen";
static vk::Instance instance;
static vk::PhysicalDevice physical_device;
static std::size_t queue_family_idx;
static vk::Device device;
static vk::Queue queue;
static vk::CommandPool cmd_pool;
static vk::CommandBuffer cmd_buffer;

// use a stack of std::functions to execute cleanup dynamically.
// to not run the risk of the cleanup stack outlasting any global handles and then calling cleanup on them, use
// std::function to store lambdas that keep their handles as copy captures as opposed to storing plain function
// pointers.
struct cleanup_stack {
    using func_t = std::function<void()>;
    std::vector<func_t> container;

    template <class FuncT>
    void push(FuncT&& func_cleanup) {
        container.emplace_back(std::forward<FuncT>(func_cleanup));
    }

    ~cleanup_stack() {
        if (!container.empty()) {
            spdlog::debug("Cleanning up.");
            std::for_each(container.crbegin(), container.crend(), [](const func_t& func_cleanup) { func_cleanup(); });
        } else {
            spdlog::debug("Cleanup stack is empty. No cleanup to be done.");
        }
    }
};
cleanup_stack cleanup;

#ifndef NDEBUG

static vk::DispatchLoaderDynamic dynamic_dispatch;
static vk::DebugUtilsMessengerEXT debug_messenger;

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT type,
                                                 const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                                 void* user_data) {

    (void)user_data;

    const auto msg_severity = vk::DebugUtilsMessageSeverityFlagBitsEXT(severity);
    const auto msg_type = vk::DebugUtilsMessageSeverityFlagBitsEXT(type);

    constexpr const auto debug_msg = "Vk Validation Layer (Type: {}): {}";
    const auto type_str = vk::to_string(msg_type);

    switch (msg_severity) {
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
        spdlog::error(debug_msg, type_str, callback_data->pMessage);
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
        spdlog::info(debug_msg, type_str, callback_data->pMessage);
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
        spdlog::trace(debug_msg, type_str, callback_data->pMessage);
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
        spdlog::warn(debug_msg, type_str, callback_data->pMessage);
        break;
    }

    return VK_FALSE;
}

static bool init_debug_messenger(vk::Instance& instance) {
    using severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using msg_type = vk::DebugUtilsMessageTypeFlagBitsEXT;
    auto sev_flags = severity::eVerbose | severity::eError | severity::eWarning;
    auto msg_flags = msg_type::eGeneral | msg_type::ePerformance | msg_type::eValidation;
    const vk::DebugUtilsMessengerCreateInfoEXT create_info(vk::DebugUtilsMessengerCreateFlagBitsEXT{}, sev_flags,
                                                           msg_flags, (PFN_vkDebugUtilsMessengerCallbackEXT)debug_cb,
                                                           nullptr);

    dynamic_dispatch = vk::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr);
    const auto opt_messenger = instance.createDebugUtilsMessengerEXT(create_info, nullptr, dynamic_dispatch);
    if (opt_messenger.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create Vk debug messenger! (VkResult: {})", vk::to_string(opt_messenger.result));
        return false;
    }

    debug_messenger = opt_messenger.value;

    // push cleanup
    cleanup.push([instance = instance, debug_messenger = debug_messenger, dynamic_dispatch = dynamic_dispatch]() {
        spdlog::debug("Destroying debug messenger.");
        instance.destroyDebugUtilsMessengerEXT(debug_messenger, nullptr, dynamic_dispatch);
    });

    return true;
}

#endif

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
        return {};
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

// vulkan related functions
static bool suitable_queue(const vk::QueueFamilyProperties& properties) {
    return properties.queueCount >= 1 && (bool)(properties.queueFlags & vk::QueueFlagBits::eCompute);
}

template <class It>
static std::optional<std::tuple<const vk::PhysicalDevice, std::size_t>>
get_suitable_device_and_queue_family_idx(It begin, It end) {
    auto start = begin;
    while (start != end) {
        const auto& device = *start;

        const auto family_properties = device.getQueueFamilyProperties();
        const auto queue_it = std::find_if(family_properties.cbegin(), family_properties.cend(), suitable_queue);
        if (queue_it != family_properties.cend()) {
            const auto queue_idx = std::distance(family_properties.cbegin(), queue_it);
            return std::tuple{device, queue_idx};
        }

        ++start;
    }

    return {};
}

static bool init_vk() {
    spdlog::debug("Creating VkInstance");

#ifndef NDEBUG
    const auto layers = {"VK_LAYER_KHRONOS_validation"};
    const auto extensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
#else
    const auto layers = {};
    const auto extensions = {};
#endif

    vk::ApplicationInfo app_info(app_name, 1, nullptr, 0);
    vk::InstanceCreateInfo inst_info({}, &app_info, layers, extensions);

    const auto opt_instance = vk::createInstance(inst_info);

    if (opt_instance.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create VkInstance! (VkResult: {})", vk::to_string(opt_instance.result));
        return false;
    }

    instance = opt_instance.value;

    // push cleanup
    cleanup.push([instance = instance]() {
        spdlog::debug("Destroying instance.");
        instance.destroy();
    });

    return true;
}

static bool init_logical_device(const std::optional<std::string>& opt_device_name) {
    spdlog::debug("Getting VkPhysicalDevice list");

    auto opt_physical_devices = instance.enumeratePhysicalDevices();

    if (opt_physical_devices.result != vk::Result::eSuccess) {
        spdlog::error("Failed to get VkPhysicalDevice list! (VkResult: {})",
                      vk::to_string(opt_physical_devices.result));
        return false;
    }

    auto& physical_devices = opt_physical_devices.value;

    spdlog::debug("Searching for suitable VkPhysicalDevice");

    auto it_end = physical_devices.end();
    if (opt_device_name) {
        const auto& device_name = *opt_device_name;

        spdlog::debug("Filtering for devices with name \"{}\"", device_name);
        it_end = std::remove_if(physical_devices.begin(), physical_devices.end(),
                                [&device_name](const vk::PhysicalDevice& cur_device) {
                                    const std::string_view name = cur_device.getProperties().deviceName;
                                    const auto find_pos = name.find(device_name);
                                    return find_pos == std::string_view::npos;
                                });

        if (it_end == physical_devices.begin()) {
            spdlog::error("Failed to find device with name \"{}\"", device_name);
            return false;
        }
    }

    const auto opt_device_queue = get_suitable_device_and_queue_family_idx(physical_devices.begin(), it_end);

    if (!opt_device_queue) {
        spdlog::error("Failed to find a suitable VkPhysicalDevice! (Requires a queue family with at least 1 queue "
                      "that supports compute shaders)");
        return false;
    }

    std::tie(physical_device, queue_family_idx) = *opt_device_queue;

    spdlog::info("Physical device: {}", physical_device.getProperties().deviceName.data());
    spdlog::debug("Queue family index: {}", queue_family_idx);

    spdlog::debug("Creating logical VkDevice");

    const auto queue_priorities = {0.f};
    const auto queue_create_info = {vk::DeviceQueueCreateInfo({}, queue_family_idx, queue_priorities)};

    const auto opt_logical_device =
        physical_device.createDevice(vk::DeviceCreateInfo({}, queue_create_info, {}, {}, {}));

    if (opt_logical_device.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create logical device! (VkResult: {})", vk::to_string(opt_logical_device.result));
        return false;
    }

    device = opt_logical_device.value;

    // push cleanup
    cleanup.push([device = device]() {
        spdlog::debug("Destroying device.");
        device.destroy();
    });

    return true;
}

static bool init_command_pool() {
    spdlog::debug("Creating VkCommandPool");

    const auto opt_cmd_pool = device.createCommandPool(
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family_idx));

    if (opt_cmd_pool.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create VkCommandPool from device! (VkResult: {})", vk::to_string(opt_cmd_pool.result));
        return false;
    }

    cmd_pool = opt_cmd_pool.value;

    // push cleanup
    cleanup.push([device = device, cmd_pool = cmd_pool]() {
        spdlog::debug("Destroying command pool.");
        device.destroyCommandPool(cmd_pool);
    });

    return true;
}

static bool init_command_buffer() {
    spdlog::debug("Creating main VkCommandBuffer");

    const auto opt_cmd_buffer =
        device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(cmd_pool, vk::CommandBufferLevel::ePrimary, 1));

    if (opt_cmd_buffer.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create VkCommandBuffer! (VkResult: {})", vk::to_string(opt_cmd_buffer.result));
        return false;
    }

    cmd_buffer = opt_cmd_buffer.value.front();

    // push cleanup
    cleanup.push([device = device, cmd_pool = cmd_pool, cmd_buffer = cmd_buffer]() {
        spdlog::debug("Freeing command buffer.");
        device.freeCommandBuffers(cmd_pool, {cmd_buffer});
    });

    return true;
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
    if (!init_vk()) {
        spdlog::critical("Failed to init Vulkan");
        return EXIT_FAILURE;
    }

#ifndef NDEBUG
    if (!init_debug_messenger(instance)) {
        spdlog::critical("Failed to init debug messenger");
        return EXIT_FAILURE;
    }
#endif

    // if list-devices is specified, list physical devices and exit
    if (list_devices) {
        const auto opt_physical_devices = instance.enumeratePhysicalDevices();
        if (opt_physical_devices.result != vk::Result::eSuccess) {
            spdlog::critical("Failed to get VkPhysicalDevice list! (VkResult: {})",
                             vk::to_string(opt_physical_devices.result));
            return EXIT_FAILURE;
        }

        const auto physical_devices = opt_physical_devices.value;

        for (const auto& cur_physical_device : physical_devices) {
            std::cout << cur_physical_device.getProperties().deviceName.data() << '\n';
        }

        return EXIT_SUCCESS;
    }

    // can begin to load image now that no other non-fatal early-exits will appear
    const auto infile = *opt_input;
    const auto fut_opt_image = std::async(open_image, infile);

    // device name
    const auto opt_device_name = argparse.present<std::string>("--device");
    if (!init_logical_device(opt_device_name)) {
        spdlog::critical("Failed to init VkDevice");
        return EXIT_FAILURE;
    }

    queue = device.getQueue(queue_family_idx, 0);

    if (!init_command_pool()) {
        spdlog::critical("Failed to init VkCommandPool");
        return EXIT_FAILURE;
    }

    if (!init_command_buffer()) {
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
