#pragma once

#include <optional>

#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.hpp>

struct vulkan_ctx {
    std::optional<vk::Instance> m_instance;
    std::optional<vk::PhysicalDevice> m_physical_device;
    std::optional<std::size_t> m_queue_family_idx;
    std::optional<vk::Device> m_device;
    std::optional<vk::Queue> m_queue;
    std::optional<vk::CommandPool> m_cmd_pool;
    std::optional<vk::CommandBuffer> m_cmd_buffer;

    bool init_instance();
    void cleanup_instance();

    bool init_logical_device(const std::optional<std::string>&);
    void cleanup_logical_device();

    bool init_command_pool();
    void cleanup_command_pool();

    bool init_command_buffer();
    void cleanup_command_buffer();

    bool list_vk_devices();

#ifndef NDEBUG
    std::optional<vk::DispatchLoaderDynamic> m_dynamic_dispatch;
    std::optional<vk::DebugUtilsMessengerEXT> m_debug_messenger;

    bool init_debug_messenger();
    void cleanup_debug_messenger();

#endif

    ~vulkan_ctx();
};
