#include "vulkan_ctx.h"

#include <iostream>

#include <spdlog/spdlog.h>
#include <vk_mem_alloc.h>

static const char* app_name = "chaq_sdfgen";

#ifndef NDEBUG

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

#endif

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

    return std::nullopt;
}

bool vulkan_ctx::init_instance() {
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

    m_instance = opt_instance.value;
    return true;
}

void vulkan_ctx::cleanup_instance() {
    if (!m_instance) return;

    spdlog::debug("Destroying instance.");
    m_instance->destroy();
}

// precondition: instance
bool vulkan_ctx::init_logical_device(const std::optional<std::string>& opt_device_name) {
    spdlog::debug("Getting VkPhysicalDevice list");
    auto opt_physical_devices = m_instance->enumeratePhysicalDevices();
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
        if (opt_device_name) {
            spdlog::error("A device name was passed in, but no devices under this name were suitable for use as per "
                          "the above error message.");
        }
        return false;
    }

    auto& [phys_dev, q_fam_idx] = *opt_device_queue;
    spdlog::info("Physical device: {}", phys_dev.getProperties().deviceName.data());
    spdlog::debug("Queue family index: {}", q_fam_idx);

    spdlog::debug("Creating logical VkDevice");
    const auto queue_priorities = {0.f};
    const auto queue_create_info = {vk::DeviceQueueCreateInfo({}, q_fam_idx, queue_priorities)};
    const auto opt_logical_device = phys_dev.createDevice(vk::DeviceCreateInfo({}, queue_create_info, {}, {}, {}));
    if (opt_logical_device.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create logical device! (VkResult: {})", vk::to_string(opt_logical_device.result));
        return false;
    }

    m_physical_device = phys_dev;
    m_queue_family_idx = q_fam_idx;
    m_device = opt_logical_device.value;
    m_queue = m_device->getQueue(*m_queue_family_idx, 0);
    return true;
}

void vulkan_ctx::cleanup_logical_device() {
    if (!m_device) return;

    spdlog::debug("Destroying device.");
    m_device->destroy();
}

// precondition: device, queue_family_idx
bool vulkan_ctx::init_command_pool() {
    spdlog::debug("Creating VkCommandPool");
    const auto opt_cmd_pool = m_device->createCommandPool(
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, *m_queue_family_idx));
    if (opt_cmd_pool.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create VkCommandPool from device! (VkResult: {})", vk::to_string(opt_cmd_pool.result));
        return false;
    }

    m_cmd_pool = opt_cmd_pool.value;
    return true;
}

void vulkan_ctx::cleanup_command_pool() {
    if (!m_device || !m_cmd_pool) return;

    spdlog::debug("Destroying command pool.");
    m_device->destroyCommandPool(*m_cmd_pool);
}

// precondition: device, cmd_pool
bool vulkan_ctx::init_command_buffer() {
    spdlog::debug("Creating main VkCommandBuffer");
    const auto opt_cmd_buffer = m_device->allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(*m_cmd_pool, vk::CommandBufferLevel::ePrimary, 1));
    if (opt_cmd_buffer.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create VkCommandBuffer! (VkResult: {})", vk::to_string(opt_cmd_buffer.result));
        return false;
    }

    m_cmd_buffer = opt_cmd_buffer.value.front();
    return true;
}

void vulkan_ctx::cleanup_command_buffer() {
    if (!m_device || !m_cmd_pool || !m_cmd_buffer) return;

    spdlog::debug("Freeing command buffer.");
    m_device->freeCommandBuffers(*m_cmd_pool, {*m_cmd_buffer});
}

bool vulkan_ctx::list_vk_devices() {
    const auto opt_physical_devices = m_instance->enumeratePhysicalDevices();
    if (opt_physical_devices.result != vk::Result::eSuccess) {
        spdlog::error("Failed to get VkPhysicalDevice list! (VkResult: {})",
                      vk::to_string(opt_physical_devices.result));
        return false;
    }

    const auto physical_devices = opt_physical_devices.value;
    for (const auto& cur_physical_device : physical_devices) {
        std::cout << cur_physical_device.getProperties().deviceName.data() << '\n';
    }

    return true;
}

// precondition: instance
bool vulkan_ctx::init_debug_messenger() {
    using severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using msg_type = vk::DebugUtilsMessageTypeFlagBitsEXT;
    auto sev_flags = severity::eVerbose | severity::eError | severity::eWarning;
    auto msg_flags = msg_type::eGeneral | msg_type::ePerformance | msg_type::eValidation;
    const vk::DebugUtilsMessengerCreateInfoEXT create_info(vk::DebugUtilsMessengerCreateFlagBitsEXT{}, sev_flags,
                                                           msg_flags, (PFN_vkDebugUtilsMessengerCallbackEXT)debug_cb,
                                                           nullptr);

    m_dynamic_dispatch = vk::DispatchLoaderDynamic(*m_instance, vkGetInstanceProcAddr);
    const auto opt_messenger = m_instance->createDebugUtilsMessengerEXT(create_info, nullptr, *m_dynamic_dispatch);
    if (opt_messenger.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create Vk debug messenger! (VkResult: {})", vk::to_string(opt_messenger.result));
        return false;
    }

    m_debug_messenger = opt_messenger.value;
    return true;
}

void vulkan_ctx::cleanup_debug_messenger() {
    if (!m_debug_messenger || !m_dynamic_dispatch) return;

    spdlog::debug("Destroying debug messenger.");
    m_instance->destroyDebugUtilsMessengerEXT(*m_debug_messenger, nullptr, *m_dynamic_dispatch);
}

vulkan_ctx::~vulkan_ctx() {
    cleanup_command_buffer();
    cleanup_command_pool();
    cleanup_logical_device();
#ifndef NDEBUG
    cleanup_debug_messenger();
#endif
    cleanup_instance();
}
