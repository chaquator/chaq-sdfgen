#include <algorithm>
#include <cstdint>
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

static const char* app_name = "chaq_sdfgen";
static vk::Instance instance;
static vk::PhysicalDevice physical_device;
static std::size_t queue_family_idx;
static vk::Device device;
static vk::Queue queue;
static vk::CommandPool cmd_pool;
static vk::CommandBuffer cmd_buffer;

template <class It>
std::optional<std::tuple<const vk::PhysicalDevice, std::size_t>> get_suitable_device_and_queue_family_idx(It begin,
                                                                                                          It end) {
    auto start = begin;
    while (start != end) {
        const auto& device = *start;

        const auto family_properties = device.getQueueFamilyProperties();
        const auto queue_it = std::find_if(
            family_properties.cbegin(), family_properties.cend(), [](const vk::QueueFamilyProperties& prop) {
                return prop.queueCount >= 1 && (bool)(prop.queueFlags & vk::QueueFlagBits::eCompute);
            });
        if (queue_it != family_properties.cend()) {
            const auto queue_idx = std::distance(family_properties.cbegin(), queue_it);
            return std::tuple{device, queue_idx};
        }

        ++start;
    }

    return {};
}

bool init_vk() {
    spdlog::trace("Creating VkInstance");

    vk::ApplicationInfo app_info(app_name, 1, nullptr, 0);
    vk::InstanceCreateInfo inst_info({}, &app_info);

    const auto instance_opt = vk::createInstance(inst_info);

    if (instance_opt.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create VkInstance! (VkResult: {})", vk::to_string(instance_opt.result));
        return false;
    }

    instance = instance_opt.value;

    return true;
}

bool init_logical_device() {
    spdlog::trace("Getting VkPhysicalDevice list");

    const auto physical_devices_opt = instance.enumeratePhysicalDevices();

    if (physical_devices_opt.result != vk::Result::eSuccess) {
        spdlog::error("Failed to get VkPhysicalDevice list! (VkResult: {})",
                      vk::to_string(physical_devices_opt.result));
        return false;
    }

    const auto& physical_devices = physical_devices_opt.value;

    spdlog::trace("Searching for suitable VkPhysicalDevice");

    const auto device_queue_opt =
        get_suitable_device_and_queue_family_idx(physical_devices.cbegin(), physical_devices.cend());

    if (!device_queue_opt) {
        spdlog::error("Failed to find a suitable VkPhysicalDevice! (Requires a queue family with at least 1 queue "
                      "that supports compute shaders)");
        return false;
    }

    std::tie(physical_device, queue_family_idx) = *device_queue_opt;

    spdlog::info("Physical device: {}", physical_device.getProperties().deviceName.data());
    spdlog::trace("Queue family index: {}", queue_family_idx);

    spdlog::trace("Creating logical VkDevice");

    const auto queue_priorities = {0.f};
    const auto queue_create_info = {vk::DeviceQueueCreateInfo({}, queue_family_idx, queue_priorities)};

    const auto logical_device_opt =
        physical_device.createDevice(vk::DeviceCreateInfo({}, queue_create_info, {}, {}, {}));

    if (logical_device_opt.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create logical device! (VkResult: {})", vk::to_string(logical_device_opt.result));
        return false;
    }

    device = logical_device_opt.value;

    return true;
}

bool init_command_pool() {
    spdlog::trace("Creating VkCommandPool");

    const auto cmd_pool_opt = device.createCommandPool(
        vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queue_family_idx));

    if (cmd_pool_opt.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create VkCommandPool from device! (VkResult: {})", vk::to_string(cmd_pool_opt.result));
        return false;
    }

    cmd_pool = cmd_pool_opt.value;

    return true;
}

bool init_command_buffer() {
    spdlog::trace("Creating main VkCommandBuffer");

    const auto cmd_buffer_opt =
        device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(cmd_pool, vk::CommandBufferLevel::ePrimary, 1));

    if (cmd_buffer_opt.result != vk::Result::eSuccess) {
        spdlog::error("Failed to create VkCommandBuffer! (VkResult: {})", vk::to_string(cmd_buffer_opt.result));
        return false;
    }

    // Should be fine even if vector is soon constructed, because vector is of handles (pointer)
    cmd_buffer = cmd_buffer_opt.value.front();

    return true;
}

void destroy_vk() {
    spdlog::trace("Cleaning up resources");

    device.freeCommandBuffers(cmd_pool, {cmd_buffer});
    device.destroyCommandPool(cmd_pool);
    device.destroy();
    instance.destroy();
}

int main() {
    spdlog::set_level(spdlog::level::trace);

    if (!init_vk()) {
        spdlog::critical("Failed to init Vulkan");
        return -1;
    };

    if (!init_logical_device()) {
        spdlog::critical("Failed to init VkDevice");
        return -1;
    }

    queue = device.getQueue(queue_family_idx, 0);

    if (!init_command_pool()) {
        spdlog::critical("Failed to init VkCommandPool");
        return -1;
    }

    if (!init_command_buffer()) {
        spdlog::critical("Failed to init VkCommandBuffer");
        return -1;
    }

    // Destroy VK
    destroy_vk();
}
