#pragma once

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <unordered_map>

namespace filetype {
enum filetype {
    png,
    jpeg,
    tga,
    bmp,
};

// Derives filetype from string by searching for it.
filetype from_str(std::string_view name, filetype fallback = png);

std::string_view to_str(filetype type);

} // namespace filetype
